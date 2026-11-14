from __future__ import annotations

import base64
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import proxy_server
import snapshot_utils

_import_data_dir = tempfile.TemporaryDirectory()
_original_data_dir = os.environ.get("VPNGATE_DATA_DIR")
os.environ["VPNGATE_DATA_DIR"] = _import_data_dir.name
try:
    import vpngate_manager as manager
finally:
    if _original_data_dir is None:
        os.environ.pop("VPNGATE_DATA_DIR", None)
    else:
        os.environ["VPNGATE_DATA_DIR"] = _original_data_dir


class FakeProcess:
    def __init__(self) -> None:
        self.running = True
        self.terminated = False

    def poll(self):
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminated = True
        self.running = False

    def wait(self, timeout=None):
        self.running = False
        return 0

    def kill(self) -> None:
        self.running = False


def valid_snapshot_rows(rows: list[tuple[str, str, str]]) -> str:
    csv_rows = [
        "#HostName,IP,Score,Ping,Speed,CountryLong,CountryShort,NumVpnSessions,OpenVPN_ConfigData_Base64"
    ]
    for index, (ip, country_long, country_short) in enumerate(rows):
        config_text = (
            "client\n"
            "dev tun\n"
            "proto udp\n"
            f"remote {ip} 1194 udp\n"
            "resolv-retry infinite\n"
            "nobind\n"
            "<ca>\nCA\n</ca>\n"
            "<cert>\nCERT\n</cert>\n"
            "<key>\nKEY\n</key>\n"
        )
        config = base64.b64encode(config_text.encode("utf-8")).decode("ascii")
        csv_rows.append(
            f"vpn{index}.example,{ip},100,20,1000,{country_long},{country_short},1,{config}"
        )
    return "\n".join(csv_rows) + "\n"


def valid_snapshot(ip: str = "198.51.100.10") -> str:
    return valid_snapshot_rows([(ip, "Japan", "JP")])


class ManagerLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.path_patches = [
            mock.patch.object(manager, "DATA_DIR", root),
            mock.patch.object(manager, "CONFIG_DIR", root / "configs"),
            mock.patch.object(manager, "NODES_FILE", root / "nodes.json"),
            mock.patch.object(manager, "STATE_FILE", root / "state.json"),
            mock.patch.object(manager, "AUTH_FILE", root / "auth.txt"),
            mock.patch.object(manager, "BLACKLIST_FILE", root / "blacklist.json"),
            mock.patch.object(manager, "US_NODE_POOL_FILE", root / "us_node_pool.json"),
            mock.patch.object(manager, "API_CACHE_FILE", root / "api_snapshot.csv"),
            mock.patch.object(manager, "API_CACHE_META_FILE", root / "api_snapshot.meta.json"),
            mock.patch.object(manager, "BUNDLED_SNAPSHOT_FILE", root / "bundled_snapshot.csv"),
            mock.patch.object(manager.vpn_utils, "DATA_DIR", root),
            mock.patch.object(manager.vpn_utils, "IP_CACHE_FILE", root / "ip_cache.json"),
        ]
        for patcher in self.path_patches:
            patcher.start()
        manager.ensure_dirs()
        manager.active_openvpn_process = None
        manager.pending_openvpn_process = None
        manager.active_openvpn_node_id = ""
        manager.active_connection_cancel_event = None
        manager.is_connecting = False
        manager.maintenance_busy = False
        manager.pending_connection_node_id = ""
        # Mirror harvesting is throttled off by default; tests that cover it re-arm it.
        manager.last_us_mirror_harvest = manager.time.time()
        manager.consecutive_proxy_failures = 0
        manager.last_proxy_failure_node_id = ""
        manager.background_refill_thread = None
        manager.background_refill_cancel_event.clear()
        manager.active_sessions.clear()

    def tearDown(self) -> None:
        if manager.connection_attempt_lock.locked():
            manager.connection_attempt_lock.release()
        manager.background_refill_cancel_event.set()
        manager.background_refill_thread = None
        for patcher in reversed(self.path_patches):
            patcher.stop()
        self.temp_dir.cleanup()

    def write_nodes(self, count: int) -> list[dict]:
        nodes = []
        for index in range(count):
            node_id = f"node-{index}"
            nodes.append(
                {
                    "id": node_id,
                    "ip": f"192.0.2.{index + 1}",
                    "remote_host": f"192.0.2.{index + 1}",
                    "remote_port": 1194,
                    "ping": index + 1,
                    "score": 1000 - index,
                    "config_text": "client\nremote 192.0.2.1 1194 udp\n",
                    "config_file": str(manager.CONFIG_DIR / f"{node_id}.ovpn"),
                    "probe_status": "not_checked",
                    "probed_at": 0,
                    "active": False,
                }
            )
        manager.write_json(manager.NODES_FILE, nodes)
        return nodes

    def test_node_probe_stops_after_target_batch(self) -> None:
        nodes = self.write_nodes(12)
        calls = []

        def fake_openvpn(config_file, **kwargs):
            calls.append(config_file)
            return True, "ready", None

        with (
            mock.patch.object(manager.vpn_utils, "ping_latency_ms", return_value=10),
            mock.patch.object(manager.vpn_utils, "enrich_ip_info"),
            mock.patch.object(manager, "run_openvpn_until_ready", side_effect=fake_openvpn),
            mock.patch.object(manager, "NODE_PROBE_WORKERS", 5),
        ):
            results = manager.test_multiple_nodes(
                [node["id"] for node in nodes],
                target_available=3,
            )

        self.assertEqual(5, len(calls))
        self.assertEqual(5, len(results))
        stored = manager.read_nodes()
        self.assertEqual(5, sum(node.get("probe_status") == "available" for node in stored))
        self.assertEqual(7, sum(node.get("probe_status") == "not_checked" for node in stored))

    def test_ip_classification_separates_proxy_use_from_network_type(self) -> None:
        residential, residential_reason = manager.vpn_utils.classify_ip_type(
            {
                "isp": "Sony Network Communications Inc.",
                "org": "Sony Network Communications Inc.",
                "proxy": True,
                "hosting": False,
                "mobile": False,
            }
        )
        softether, softether_reason = manager.vpn_utils.classify_ip_type(
            {
                "isp": "SoftEther",
                "org": "SoftEther Corporation",
                "proxy": True,
                "hosting": False,
                "mobile": False,
            }
        )
        hosting, hosting_reason = manager.vpn_utils.classify_ip_type(
            {"proxy": True, "hosting": True, "mobile": False}
        )
        mobile, mobile_reason = manager.vpn_utils.classify_ip_type(
            {"proxy": False, "hosting": False, "mobile": True}
        )
        unknown, unknown_reason = manager.vpn_utils.classify_ip_type(
            {"proxy": True, "hosting": False, "mobile": False}
        )

        self.assertEqual(("residential", "consumer_or_unclassified_network"), (residential, residential_reason))
        self.assertEqual(("hosting", "proxy_provider_datacenter"), (softether, softether_reason))
        self.assertEqual(("hosting", "hosting_flag"), (hosting, hosting_reason))
        self.assertEqual(("mobile", "mobile_flag"), (mobile, mobile_reason))
        self.assertEqual(("unknown", "missing_provider_data"), (unknown, unknown_reason))
        self.assertEqual("low", manager.vpn_utils.classification_confidence(unknown_reason))

    def test_ip_enrichment_reclassifies_legacy_cache_and_keeps_proxy_quality(self) -> None:
        ip = "118.240.250.95"
        manager.vpn_utils.IP_CACHE_FILE.write_text(
            json.dumps(
                {
                    ip: {
                        "ip_type": "hosting",
                        "quality": "proxy",
                        "cached_at": 9999999999,
                        "classification_version": 1,
                    }
                }
            ),
            encoding="utf-8",
        )
        api_result = [
            {
                "status": "success",
                "query": ip,
                "country": "Japan",
                "regionName": "Tokyo",
                "city": "Tokyo",
                "isp": "Sony Network Communications Inc.",
                "org": "Sony Network Communications Inc.",
                "as": "AS2527 Sony Network Communications Inc.",
                "asname": "Sony Network Communications Inc.",
                "proxy": True,
                "hosting": False,
                "mobile": False,
            }
        ]
        primary = mock.MagicMock()
        primary.read.return_value = json.dumps(api_result).encode("utf-8")
        primary.__enter__.return_value = primary
        # ipapi.is now supplements every classified IP with cleanliness signals.
        secondary = mock.MagicMock()
        secondary.read.return_value = json.dumps(
            {
                "is_datacenter": False,
                "is_mobile": False,
                "is_vpn": False,
                "is_proxy": False,
                "is_tor": False,
                "is_abuser": False,
                "company": {"abuser_score": "0.0 (Very Low)"},
            }
        ).encode("utf-8")
        secondary.__enter__.return_value = secondary
        node = {"id": "sony", "ip": ip}

        with mock.patch.object(
            manager.vpn_utils.urllib.request,
            "urlopen",
            side_effect=[primary, secondary],
        ) as urlopen_mock:
            manager.vpn_utils.enrich_ip_info([node])

        self.assertEqual("residential", node["ip_type"])
        self.assertEqual("proxy", node["quality"])
        self.assertTrue(node["is_proxy"])
        self.assertFalse(node["is_hosting"])
        self.assertEqual(["ip-api.com", "ipapi.is"], node["ip_type_sources"])
        self.assertEqual(2, urlopen_mock.call_count)
        cache = json.loads(manager.vpn_utils.IP_CACHE_FILE.read_text(encoding="utf-8"))
        self.assertEqual(manager.vpn_utils.IP_CLASSIFICATION_VERSION, cache[ip]["classification_version"])

    def test_ambiguous_datacenter_uses_secondary_source_and_geo_country(self) -> None:
        ip = "219.100.37.98"
        primary_payload = [{
            "status": "success",
            "query": ip,
            "country": "Japan",
            "countryCode": "JP",
            "regionName": "Tokyo",
            "city": "Chiyoda",
            "isp": "SoftEther",
            "org": "SoftEther Corporation",
            "as": "AS36599 SoftEther",
            "asname": "SOFTETHER",
            "proxy": True,
            "hosting": False,
            "mobile": False,
        }]
        primary = mock.MagicMock()
        primary.read.return_value = json.dumps(primary_payload).encode("utf-8")
        primary.__enter__.return_value = primary
        secondary = mock.MagicMock()
        secondary.read.return_value = json.dumps({"is_datacenter": True, "is_vpn": True}).encode("utf-8")
        secondary.__enter__.return_value = secondary
        node = {"id": "softether", "ip": ip}

        with mock.patch.object(
            manager.vpn_utils.urllib.request,
            "urlopen",
            side_effect=[primary, secondary],
        ):
            manager.vpn_utils.enrich_ip_info([node])

        self.assertEqual("hosting", node["ip_type"])
        self.assertEqual("high", node["ip_type_confidence"])
        self.assertEqual("datacenter", node["quality"])
        self.assertTrue(node["is_hosting"])
        self.assertEqual(["ip-api.com", "ipapi.is"], node["ip_type_sources"])
        self.assertEqual("JP", node["geo_country_short"])

    def test_unverified_datacenter_conflict_becomes_unknown(self) -> None:
        ip = "203.0.113.10"
        primary_payload = [{
            "status": "success",
            "query": ip,
            "country": "Japan",
            "countryCode": "JP",
            "regionName": "Tokyo",
            "city": "Tokyo",
            "isp": "Example VPS",
            "org": "Example VPS Hosting",
            "as": "AS64500 Example",
            "asname": "EXAMPLE",
            "proxy": True,
            "hosting": False,
            "mobile": False,
        }]
        primary = mock.MagicMock()
        primary.read.return_value = json.dumps(primary_payload).encode("utf-8")
        primary.__enter__.return_value = primary
        node = {"id": "ambiguous", "ip": ip}

        with mock.patch.object(
            manager.vpn_utils.urllib.request,
            "urlopen",
            side_effect=[primary, TimeoutError("secondary unavailable")],
        ):
            manager.vpn_utils.enrich_ip_info([node])

        self.assertEqual("unknown", node["ip_type"])
        self.assertEqual("low", node["ip_type_confidence"])
        strict = manager.apply_routing_filters([node], {"routing_mode": "auto", "routing_ip_type": "residential"})
        self.assertEqual([], strict)

    def test_missing_provider_data_uses_secondary_source_or_stays_unknown(self) -> None:
        ip = "203.0.113.11"
        primary_payload = [{
            "status": "success",
            "query": ip,
            "country": "Japan",
            "countryCode": "JP",
            "regionName": "Tokyo",
            "city": "Tokyo",
            "isp": "",
            "org": "",
            "as": "",
            "asname": "",
            "proxy": True,
            "hosting": False,
            "mobile": False,
        }]
        primary = mock.MagicMock()
        primary.read.return_value = json.dumps(primary_payload).encode("utf-8")
        primary.__enter__.return_value = primary
        node = {"id": "missing-provider", "ip": ip}

        with mock.patch.object(
            manager.vpn_utils.urllib.request,
            "urlopen",
            side_effect=[primary, TimeoutError("secondary unavailable")],
        ):
            manager.vpn_utils.enrich_ip_info([node])

        self.assertEqual("unknown", node["ip_type"])
        self.assertEqual("provider_data_unverified", node["ip_type_reason"])
        self.assertEqual("low", node["ip_type_confidence"])
        strict = manager.apply_routing_filters(
            [node],
            {"routing_mode": "auto", "routing_ip_type": "residential"},
        )
        self.assertEqual([], strict)

    def test_strict_residential_filter_requires_medium_or_high_confidence(self) -> None:
        nodes = [
            {"id": "low", "ip_type": "residential", "ip_type_confidence": "low"},
            {"id": "medium", "ip_type": "residential", "ip_type_confidence": "medium"},
            {"id": "mobile", "ip_type": "mobile", "ip_type_confidence": "high"},
            {"id": "hosting", "ip_type": "hosting", "ip_type_confidence": "high"},
        ]

        strict = manager.apply_routing_filters(
            nodes,
            {"routing_mode": "auto", "routing_ip_type": "residential"},
        )

        self.assertEqual(["medium", "mobile"], [node["id"] for node in strict])

    def test_risk_control_index_orders_ip_types_from_safe_to_risky(self) -> None:
        mobile = manager.risk_control_index({"ip_type": "mobile"})
        residential = manager.risk_control_index({"ip_type": "residential"})
        unknown = manager.risk_control_index({"ip_type": "unknown"})
        empty = manager.risk_control_index({"ip_type": ""})
        hosting = manager.risk_control_index({"ip_type": "hosting"})

        self.assertLess(mobile, residential)
        self.assertLess(residential, unknown)
        self.assertEqual(unknown, empty)
        self.assertLess(unknown, hosting)

        # A network flagged as a public proxy is riskier than the same type
        # without the flag, but still ordered by its underlying network type.
        residential_proxy = manager.risk_control_index(
            {"ip_type": "residential", "is_proxy": True}
        )
        self.assertGreater(residential_proxy, residential)
        self.assertLess(residential_proxy, hosting)

    def test_risk_control_index_penalizes_unclean_ip_signals(self) -> None:
        clean = manager.risk_control_index({"ip_type": "residential"})
        proxy = manager.risk_control_index({"ip_type": "residential", "is_proxy": True})
        vpn = manager.risk_control_index({"ip_type": "residential", "is_vpn": True})
        abuser = manager.risk_control_index({"ip_type": "residential", "is_abuser": True})
        tor = manager.risk_control_index({"ip_type": "residential", "is_tor": True})
        reputation = manager.risk_control_index(
            {"ip_type": "residential", "abuser_score": "1.0 (Very High)"}
        )

        # Every "dirty" signal makes a residential IP riskier than a clean one.
        self.assertLess(clean, proxy)
        self.assertLess(proxy, vpn)
        self.assertLess(vpn, abuser)
        self.assertLess(abuser, tor)
        # The continuous abuser reputation from ipapi.is also raises the index.
        self.assertGreater(reputation, clean)

    def test_enrich_supplements_every_ip_with_ipapi_is_cleanliness(self) -> None:
        ip = "198.51.100.7"
        primary_payload = [{
            "status": "success",
            "query": ip,
            "country": "United States",
            "countryCode": "US",
            "regionName": "California",
            "city": "Los Angeles",
            "isp": "Comcast Cable",
            "org": "Comcast Cable Communications",
            "as": "AS7922 Comcast",
            "asname": "COMCAST",
            "proxy": False,
            "hosting": False,
            "mobile": False,
        }]
        primary = mock.MagicMock()
        primary.read.return_value = json.dumps(primary_payload).encode("utf-8")
        primary.__enter__.return_value = primary
        # ip-api.com sees a clean residential IP, but ipapi.is flags it as a
        # public VPN / known abuser — the risk index must reflect that.
        secondary = mock.MagicMock()
        secondary.read.return_value = json.dumps(
            {
                "is_datacenter": False,
                "is_mobile": False,
                "is_vpn": True,
                "is_proxy": False,
                "is_tor": False,
                "is_abuser": True,
                "company": {"abuser_score": "0.85 (High)"},
            }
        ).encode("utf-8")
        secondary.__enter__.return_value = secondary
        node = {"id": "us-home", "ip": ip}

        with mock.patch.object(
            manager.vpn_utils.urllib.request,
            "urlopen",
            side_effect=[primary, secondary],
        ):
            manager.vpn_utils.enrich_ip_info([node])

        self.assertEqual("residential", node["ip_type"])
        self.assertTrue(node["is_vpn"])
        self.assertTrue(node["is_abuser"])
        self.assertTrue(node["is_proxy"])  # VPN flag implies proxy-like exit
        self.assertEqual(["ip-api.com", "ipapi.is"], node["ip_type_sources"])
        clean_home = manager.risk_control_index({"ip_type": "residential"})
        self.assertEqual(node["risk_control_index"], manager.risk_control_index(node))
        self.assertGreater(node["risk_control_index"], clean_home)

    def test_available_nodes_sorted_by_lowest_risk_control_index_first(self) -> None:
        nodes = [
            {"id": "hosting-fast", "probe_status": "available", "ip_type": "hosting",
             "latency_ms": 10, "score": 500},
            {"id": "residential-slow", "probe_status": "available", "ip_type": "residential",
             "latency_ms": 300, "score": 100},
            {"id": "mobile-mid", "probe_status": "available", "ip_type": "mobile",
             "latency_ms": 150, "score": 100},
        ]

        ordered = manager.sort_all_nodes(nodes)

        # Lowest risk-control index wins even when a riskier node is faster.
        self.assertEqual(
            ["mobile-mid", "residential-slow", "hosting-fast"],
            [node["id"] for node in ordered],
        )

    def test_us_node_priority_ranks_us_residential_then_other_us_then_rest(self) -> None:
        us_home = {"country_short": "US", "ip_type": "residential", "ip_type_confidence": "medium"}
        self.assertEqual(manager.US_RESIDENTIAL_TIER, manager.us_node_priority(us_home))
        self.assertTrue(manager.is_us_residential_node(us_home))

        # Other US nodes stay ahead of every non-US node, but behind US residential.
        for other_us in (
            {"country_short": "us", "ip_type": "hosting", "ip_type_confidence": "high"},
            {"country_short": "US", "ip_type": "mobile", "ip_type_confidence": "high"},
            {"country_short": "US"},
            # An unverified "residential" guess is not trusted as home broadband.
            {"country_short": "US", "ip_type": "residential", "ip_type_confidence": "low"},
            {"country_short": "US", "ip_type": "residential"},
        ):
            self.assertEqual(manager.US_OTHER_TIER, manager.us_node_priority(other_us), other_us)
            self.assertTrue(manager.is_us_node(other_us))

        self.assertEqual(
            manager.NON_US_TIER,
            manager.us_node_priority(
                {"country_short": "JP", "ip_type": "residential", "ip_type_confidence": "high"}
            ),
        )

    def test_us_node_requires_ip_intel_not_to_contradict_declared_country(self) -> None:
        declared_us_located_elsewhere = {
            "country_short": "US", "geo_country_short": "CA",
            "ip_type": "residential", "ip_type_confidence": "medium",
        }
        self.assertFalse(manager.is_us_node(declared_us_located_elsewhere))
        self.assertEqual(manager.NON_US_TIER, manager.us_node_priority(declared_us_located_elsewhere))
        self.assertTrue(manager.is_us_node({"country_short": "US", "geo_country_short": "us"}))

    def test_us_residential_wins_over_faster_us_hosting_and_cleaner_non_us(self) -> None:
        nodes = [
            {"id": "jp-mobile", "probe_status": "available", "country_short": "JP",
             "ip_type": "mobile", "ip_type_confidence": "high", "latency_ms": 5, "score": 900},
            {"id": "us-hosting-fast", "probe_status": "available", "country_short": "US",
             "ip_type": "hosting", "ip_type_confidence": "high", "latency_ms": 10, "score": 900},
            {"id": "us-mobile", "probe_status": "available", "country_short": "US",
             "ip_type": "mobile", "ip_type_confidence": "high", "latency_ms": 20, "score": 900},
            {"id": "us-home-slow", "probe_status": "available", "country_short": "US",
             "ip_type": "residential", "ip_type_confidence": "medium", "latency_ms": 400, "score": 1},
        ]

        ordered = manager.sort_all_nodes(nodes)

        self.assertEqual(
            ["us-home-slow", "us-mobile", "us-hosting-fast", "jp-mobile"],
            [node["id"] for node in ordered],
        )

    def test_probe_priority_tests_us_residential_before_other_us_nodes(self) -> None:
        nodes = [
            {"id": "jp-fast", "country_short": "JP", "ping": 1},
            {"id": "us-hosting", "country_short": "US", "ping": 2,
             "ip_type": "hosting", "ip_type_confidence": "high"},
            {"id": "us-unknown", "country_short": "US", "ping": 3},
            {"id": "us-home", "country_short": "US", "ping": 90,
             "ip_type": "residential", "ip_type_confidence": "medium"},
        ]

        ordered = sorted(nodes, key=manager.probe_priority_key)

        self.assertEqual(
            ["us-home", "us-unknown", "us-hosting", "jp-fast"],
            [n["id"] for n in ordered],
        )

    def test_us_node_wins_over_lower_risk_but_non_us_node(self) -> None:
        nodes = [
            # Lowest risk-control index overall, but not a US node.
            {"id": "jp-mobile", "probe_status": "available", "country_short": "JP",
             "ip_type": "mobile", "latency_ms": 10, "score": 500},
            # US hosting but slower/riskier — must still come first.
            {"id": "us-hosting", "probe_status": "available", "country_short": "US",
             "ip_type": "hosting", "latency_ms": 300, "score": 100},
            # US node that is not available must not outrank available nodes.
            {"id": "us-down", "probe_status": "unavailable", "country_short": "US",
             "ip_type": "residential", "latency_ms": 5, "score": 900},
        ]

        ordered = manager.sort_all_nodes(nodes)

        self.assertEqual(
            ["us-hosting", "jp-mobile", "us-down"],
            [node["id"] for node in ordered],
        )

    def test_probe_priority_puts_us_nodes_before_lower_ping_nodes(self) -> None:
        nodes = [
            {"id": "jp-fast", "country_short": "JP", "ping": 2, "score": 900},
            {"id": "kr-fast", "country_short": "KR", "ping": 3, "score": 900},
            {"id": "us-slower", "country_short": "US", "ping": 40, "score": 100},
        ]

        ordered = sorted(nodes, key=manager.probe_priority_key)

        self.assertEqual(["us-slower", "jp-fast", "kr-fast"], [n["id"] for n in ordered])

    def test_auto_switch_probes_untested_us_node_before_choosing_other_country(self) -> None:
        pool = [
            {"id": "jp", "country_short": "JP", "ip_type": "mobile", "latency_ms": 10,
             "score": 5, "probe_status": "available"},
            {"id": "us", "country_short": "US", "ping": 8, "probe_status": "not_checked"},
        ]
        connected: list[str] = []

        def fake_probe(ids, target_available=None):
            for node in pool:
                if node["id"] in ids:
                    node.update(probe_status="available", latency_ms=80)
            return [{"id": i, "probe_status": "available", "probe_message": "ok"} for i in ids]

        with mock.patch.object(manager, "read_nodes", lambda: [dict(n) for n in pool]), \
                mock.patch.object(manager, "load_ui_config",
                                  lambda: {"connection_enabled": True, "routing_mode": "auto"}), \
                mock.patch.object(manager, "test_multiple_nodes", fake_probe), \
                mock.patch.object(manager, "log_to_json", lambda *a, **k: None), \
                mock.patch.object(manager, "connect_node", connected.append):
            manager.auto_switch_node()

        self.assertEqual(["us"], connected)

    def _run_prefer_us(self, pool, *, cooldown_elapsed=10**6, connect_error=None):
        connected: list[str] = []
        recovered: list[str] = []

        def fake_connect(node_id):
            if connect_error:
                raise connect_error
            connected.append(node_id)

        with mock.patch.object(manager, "read_nodes", lambda: [dict(n) for n in pool]), \
                mock.patch.object(manager, "load_ui_config",
                                  lambda: {"connection_enabled": True, "routing_mode": "auto"}), \
                mock.patch.object(manager, "active_openvpn_running", lambda: True), \
                mock.patch.object(manager, "log_to_json", lambda *a, **k: None), \
                mock.patch.object(manager, "connect_node", fake_connect), \
                mock.patch.object(manager, "recover_after_manual_connect_failure", recovered.append), \
                mock.patch.object(manager, "last_prefer_us_switch_attempt",
                                  manager.time.time() - cooldown_elapsed):
            manager.maybe_switch_to_us_node()
        return connected, recovered

    def test_prefer_us_switches_from_non_us_active_node(self) -> None:
        pool = [
            {"id": "jp", "country_short": "JP", "active": True, "probe_status": "available"},
            {"id": "us", "country_short": "US", "probe_status": "available", "latency_ms": 90},
        ]
        connected, _ = self._run_prefer_us(pool)
        self.assertEqual(["us"], connected)

    def test_prefer_us_keeps_us_active_node_and_respects_cooldown(self) -> None:
        us_active = [
            {"id": "us1", "country_short": "US", "active": True, "probe_status": "available"},
            {"id": "us2", "country_short": "US", "probe_status": "available"},
        ]
        self.assertEqual([], self._run_prefer_us(us_active)[0])

        jp_active = [
            {"id": "jp", "country_short": "JP", "active": True, "probe_status": "available"},
            {"id": "us", "country_short": "US", "probe_status": "available"},
        ]
        self.assertEqual([], self._run_prefer_us(jp_active, cooldown_elapsed=5)[0])

    def test_prefer_us_upgrades_from_us_hosting_to_us_residential(self) -> None:
        pool = [
            {"id": "us-dc", "country_short": "US", "active": True, "probe_status": "available",
             "ip_type": "hosting", "ip_type_confidence": "high"},
            {"id": "us-dc-2", "country_short": "US", "probe_status": "available", "latency_ms": 5,
             "ip_type": "hosting", "ip_type_confidence": "high"},
            {"id": "us-home", "country_short": "US", "probe_status": "available", "latency_ms": 200,
             "ip_type": "residential", "ip_type_confidence": "medium"},
        ]
        connected, _ = self._run_prefer_us(pool)
        self.assertEqual(["us-home"], connected)

    def test_prefer_us_keeps_active_us_residential_node(self) -> None:
        pool = [
            {"id": "us-home", "country_short": "US", "active": True, "probe_status": "available",
             "ip_type": "residential", "ip_type_confidence": "medium"},
            {"id": "us-home-2", "country_short": "US", "probe_status": "available", "latency_ms": 1,
             "ip_type": "residential", "ip_type_confidence": "high"},
        ]
        self.assertEqual([], self._run_prefer_us(pool)[0])

    def test_prefer_us_does_not_treat_busy_switch_as_node_failure(self) -> None:
        pool = [
            {"id": "jp", "country_short": "JP", "active": True, "probe_status": "available"},
            {"id": "us", "country_short": "US", "probe_status": "available"},
        ]
        connected, recovered = self._run_prefer_us(
            pool, connect_error=manager.ConnectionBusy("busy")
        )
        self.assertEqual([], connected)
        self.assertEqual([], recovered)

    def test_prefer_us_recovers_previous_node_when_switch_fails(self) -> None:
        pool = [
            {"id": "jp", "country_short": "JP", "active": True, "probe_status": "available"},
            {"id": "us", "country_short": "US", "probe_status": "available"},
        ]
        connected, recovered = self._run_prefer_us(pool, connect_error=RuntimeError("boom"))
        self.assertEqual([], connected)
        self.assertEqual(["jp"], recovered)

    def test_background_ip_enrichment_merges_metadata_without_replacing_status(self) -> None:
        nodes = self.write_nodes(2)
        nodes[0]["probe_status"] = "available"
        manager.write_json(manager.NODES_FILE, nodes)

        def fake_enrich(items):
            for item in items:
                item["ip_type"] = "residential"
                item["quality"] = "proxy"
                item["owner"] = "Consumer ISP"
                item["is_proxy"] = True

        with mock.patch.object(manager.vpn_utils, "enrich_ip_info", side_effect=fake_enrich):
            changed = manager.enrich_stored_nodes()

        stored = manager.read_nodes()
        self.assertGreater(changed, 0)
        self.assertEqual("available", next(node for node in stored if node["id"] == "node-0")["probe_status"])
        self.assertTrue(all(node["ip_type"] == "residential" for node in stored))

    def test_source_deadline_still_tries_official_http(self) -> None:
        csv_text = valid_snapshot()

        def fake_fetch(url, verify_ssl=True, deadline_seconds=None):
            if url == manager.API_HTTPS_URL:
                raise manager.SourceDeadlineExceeded("slow official source")
            if url == manager.API_HTTP_URL:
                return csv_text
            raise AssertionError(f"unexpected source: {url}")

        with (
            mock.patch.object(manager, "fetch_api_text_with_deadline", side_effect=fake_fetch) as fetch_mock,
            mock.patch.object(manager, "load_blacklist", return_value={}),
            mock.patch.object(manager, "log_to_json"),
        ):
            nodes = manager.fetch_candidates()

        self.assertEqual(1, len(nodes))
        self.assertEqual(
            [manager.API_HTTPS_URL, manager.API_HTTP_URL],
            [call.args[0] for call in fetch_mock.call_args_list],
        )

    def test_probe_failure_preserves_existing_ip_metadata(self) -> None:
        nodes = self.write_nodes(1)
        nodes[0].update(
            {
                "owner": "Existing ISP",
                "location": "日本 东京",
                "ip_type": "residential",
                "ip_type_confidence": "medium",
            }
        )
        manager.write_json(manager.NODES_FILE, nodes)

        with (
            mock.patch.object(manager.vpn_utils, "ping_latency_ms", return_value=0),
            mock.patch.object(manager, "run_openvpn_until_ready", return_value=(False, "offline", None)),
        ):
            manager.test_multiple_nodes([nodes[0]["id"]])

        stored = manager.read_nodes()[0]
        self.assertEqual("unavailable", stored["probe_status"])
        self.assertEqual("Existing ISP", stored["owner"])
        self.assertEqual("日本 东京", stored["location"])
        self.assertEqual("residential", stored["ip_type"])
        self.assertEqual("medium", stored["ip_type_confidence"])

    def test_country_matching_accepts_iso_and_legacy_name(self) -> None:
        node = {"country": "日本", "country_short": "JP"}
        self.assertTrue(manager.country_matches(node["country"], "JP", node["country_short"]))
        self.assertTrue(manager.country_matches(node["country"], "日本", node["country_short"]))
        self.assertFalse(manager.country_matches(node["country"], "KR", node["country_short"]))
        self.assertEqual("JP", manager.normalize_routing_country("日本", [node]))

    def test_web_and_proxy_ports_must_be_distinct(self) -> None:
        self.assertTrue(manager.ports_conflict(8787, "8787"))
        self.assertFalse(manager.ports_conflict(8787, 7928))

    def test_ui_connection_requires_tunnel_and_proxy_readiness(self) -> None:
        manager.active_openvpn_node_id = "node-1"
        manager.active_openvpn_process = FakeProcess()
        base_state = {"is_connecting": False, "tunnel_ready": True, "proxy_ready": False, "proxy_ok": False}
        self.assertFalse(manager.connection_ready_for_ui(base_state))
        ready_state = {**base_state, "proxy_ready": True, "proxy_ok": True}
        self.assertTrue(manager.connection_ready_for_ui(ready_state))

    def test_manual_disconnect_state_clears_all_readiness_flags(self) -> None:
        nodes = self.write_nodes(1)
        nodes[0]["active"] = True
        manager.write_json(manager.NODES_FILE, nodes)
        manager.set_state(
            is_connecting=True,
            tunnel_ready=True,
            proxy_ready=True,
            proxy_ok=True,
            proxy_ip="198.51.100.20",
        )

        with mock.patch.object(manager, "stop_active_openvpn") as stop_mock:
            manager.clear_active_connection_state("手动断开连接")

        stop_mock.assert_called_once_with()
        state = manager.get_state()
        self.assertFalse(state["is_connecting"])
        self.assertFalse(state["tunnel_ready"])
        self.assertFalse(state["proxy_ready"])
        self.assertFalse(state["proxy_ok"])
        self.assertEqual("-", state["proxy_ip"])
        self.assertFalse(any(node.get("active") for node in manager.read_nodes()))

    def test_ui_auth_json_is_written_private(self) -> None:
        auth_file = manager.DATA_DIR / "ui_auth.json"
        manager.write_json(auth_file, {"username": "test", "password": "secret"})
        if os.name != "nt":
            self.assertEqual(0o600, stat.S_IMODE(auth_file.stat().st_mode))

    def test_ui_host_defaults_to_loopback(self) -> None:
        auth_file = manager.DATA_DIR / "ui_auth.json"
        auth_file.unlink(missing_ok=True)
        try:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("UI_HOST", None)
                config = manager.load_ui_config()
            self.assertEqual("127.0.0.1", config["host"])
        finally:
            auth_file.unlink(missing_ok=True)

    def test_persisted_wildcard_ui_host_is_hardened_to_loopback(self) -> None:
        auth_file = manager.DATA_DIR / "ui_auth.json"
        try:
            for wildcard in ("::", "0.0.0.0", ""):
                manager.write_json(
                    auth_file,
                    {"username": "u", "password": "p", "host": wildcard, "port": 8787},
                )
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("UI_HOST", None)
                    config = manager.load_ui_config()
                self.assertEqual(
                    "127.0.0.1", config["host"], f"wildcard {wildcard!r} not hardened"
                )
        finally:
            auth_file.unlink(missing_ok=True)

    def test_explicit_external_ui_host_env_is_respected(self) -> None:
        auth_file = manager.DATA_DIR / "ui_auth.json"
        try:
            manager.write_json(
                auth_file,
                {"username": "u", "password": "p", "host": "0.0.0.0", "port": 8787},
            )
            with mock.patch.dict(os.environ, {"UI_HOST": "0.0.0.0"}, clear=False):
                config = manager.load_ui_config()
            self.assertEqual("0.0.0.0", config["host"])
        finally:
            auth_file.unlink(missing_ok=True)

    def test_source_deadline_limits_total_fetch_time(self) -> None:
        def slow_fetch(url, verify_ssl=True):
            threading.Event().wait(0.1)
            return valid_snapshot()

        with mock.patch.object(manager, "fetch_api_text", side_effect=slow_fetch):
            started = manager.time.monotonic()
            with self.assertRaises(manager.SourceDeadlineExceeded):
                manager.fetch_api_text_with_deadline(
                    manager.API_HTTPS_URL,
                    deadline_seconds=0.01,
                )

        self.assertLess(manager.time.monotonic() - started, 0.08)

    def test_node_probe_stops_after_systemic_openvpn_failure(self) -> None:
        nodes = self.write_nodes(12)

        with (
            mock.patch.object(manager.vpn_utils, "ping_latency_ms", return_value=10),
            mock.patch.object(
                manager,
                "run_openvpn_until_ready",
                return_value=(False, "[ERR_OVPN_TUN_NOT_AVAILABLE] missing TUN", None),
            ) as openvpn_mock,
            mock.patch.object(manager, "NODE_PROBE_WORKERS", 5),
            mock.patch.object(manager, "log_to_json"),
        ):
            results = manager.test_multiple_nodes(
                [node["id"] for node in nodes],
                target_available=3,
            )

        self.assertEqual(5, openvpn_mock.call_count)
        self.assertEqual(5, len(results))
        stored = manager.read_nodes()
        self.assertEqual(5, sum(node.get("probe_status") == "unavailable" for node in stored))
        self.assertEqual(7, sum(node.get("probe_status") == "not_checked" for node in stored))

    def test_maintenance_does_not_start_second_batch_after_systemic_failure(self) -> None:
        candidates = self.write_nodes(12)

        with (
            mock.patch.object(manager, "fetch_candidates", return_value=candidates),
            mock.patch.object(manager.vpn_utils, "ping_latency_ms", return_value=10),
            mock.patch.object(
                manager,
                "run_openvpn_until_ready",
                return_value=(False, "[ERR_OVPN_CMD_NOT_FOUND] openvpn missing", None),
            ) as openvpn_mock,
            mock.patch.object(manager, "NODE_PROBE_WORKERS", 5),
            mock.patch.object(manager, "log_to_json"),
        ):
            result = manager.maintain_valid_nodes()

        self.assertEqual(5, openvpn_mock.call_count)
        self.assertIn("Tested 5", result)

    def test_cancel_pending_connection_stops_handshake_process(self) -> None:
        process = FakeProcess()
        event = threading.Event()
        manager.pending_openvpn_process = process
        manager.active_connection_cancel_event = event
        manager.is_connecting = True
        previous_epoch = manager.connection_epoch

        manager.cancel_pending_connection_attempt()

        self.assertTrue(event.is_set())
        self.assertTrue(process.terminated)
        self.assertIsNone(manager.pending_openvpn_process)
        self.assertFalse(manager.is_connecting)
        self.assertEqual(previous_epoch + 1, manager.connection_epoch)

    def test_proxy_failures_reset_when_node_changes(self) -> None:
        self.assertEqual(1, manager.record_proxy_failure("node-a"))
        self.assertEqual(2, manager.record_proxy_failure("node-a"))
        self.assertEqual(1, manager.record_proxy_failure("node-b"))
        manager.reset_proxy_failure_counter("node-b")
        self.assertEqual(1, manager.record_proxy_failure("node-b"))

    def test_failed_switch_preflight_keeps_current_connection(self) -> None:
        nodes = self.write_nodes(2)
        nodes[0]["active"] = True
        manager.write_json(manager.NODES_FILE, nodes)
        current_process = FakeProcess()
        manager.active_openvpn_process = current_process
        manager.active_openvpn_node_id = nodes[0]["id"]

        with (
            mock.patch.object(
                manager,
                "run_openvpn_until_ready",
                return_value=(False, "preflight failed", None),
            ),
            mock.patch.object(manager, "log_to_json"),
        ):
            with self.assertRaisesRegex(RuntimeError, "已保留当前连接"):
                manager.connect_node(nodes[1]["id"])

        self.assertIs(manager.active_openvpn_process, current_process)
        self.assertEqual(nodes[0]["id"], manager.active_openvpn_node_id)
        self.assertTrue(current_process.running)
        stored = {node["id"]: node for node in manager.read_nodes()}
        self.assertEqual("unavailable", stored[nodes[1]["id"]]["probe_status"])

    def test_proxy_failure_does_not_report_connection_success(self) -> None:
        nodes = self.write_nodes(1)
        process = FakeProcess()

        with (
            mock.patch.object(
                manager,
                "run_openvpn_until_ready",
                return_value=(True, "ready", process),
            ),
            mock.patch.object(manager, "setup_policy_routing", return_value=False),
            mock.patch.object(manager, "cleanup_policy_routing"),
            mock.patch.object(manager.vpn_utils, "ping_latency_ms", return_value=10),
            mock.patch.object(manager, "check_proxy_health", return_value={"ok": False, "error": "no route"}),
            mock.patch.object(manager, "log_to_json"),
        ):
            with self.assertRaisesRegex(RuntimeError, "代理出口不可用"):
                manager.connect_node(nodes[0]["id"])

        self.assertFalse(process.running)
        self.assertIsNone(manager.active_openvpn_process)
        self.assertEqual("", manager.active_openvpn_node_id)
        stored = manager.read_nodes()
        self.assertEqual("unavailable", stored[0]["probe_status"])

    def test_manual_failure_recovery_prefers_previous_node(self) -> None:
        with (
            mock.patch.object(manager, "active_openvpn_running", return_value=False),
            mock.patch.object(manager, "connect_node", return_value="connected") as connect_mock,
            mock.patch.object(manager, "log_to_json"),
            mock.patch.object(manager, "auto_switch_node") as auto_switch_mock,
        ):
            manager.recover_after_manual_connect_failure("old-node")

        connect_mock.assert_called_once_with("old-node")
        auto_switch_mock.assert_not_called()

    def test_auto_switch_exhaustion_schedules_background_refill(self) -> None:
        with (
            mock.patch.object(manager, "schedule_background_refill", return_value=True) as schedule_mock,
            mock.patch.object(manager, "log_to_json") as log_mock,
        ):
            manager.auto_switch_node(attempt=manager.AUTO_SWITCH_MAX_ATTEMPTS)

        schedule_mock.assert_called_once_with()
        log_mock.assert_called_once_with("INFO", "Main", "连续自动切换失败，已启动唯一后台节点补齐任务")

    def test_physical_interface_detection_is_cached(self) -> None:
        original_cache = manager.vpn_utils.physical_interface_cache
        manager.vpn_utils.physical_interface_cache = (None, 0.0)
        try:
            with mock.patch.object(
                manager.vpn_utils,
                "_detect_physical_interface",
                return_value="eth0",
            ) as detect_mock:
                self.assertEqual("eth0", manager.vpn_utils.get_physical_interface())
                self.assertEqual("eth0", manager.vpn_utils.get_physical_interface())
            detect_mock.assert_called_once_with()
        finally:
            manager.vpn_utils.physical_interface_cache = original_cache

    def test_forced_refresh_keeps_healthy_active_connection(self) -> None:
        process = FakeProcess()
        manager.active_openvpn_process = process
        manager.active_openvpn_node_id = "active-node"

        with (
            mock.patch.object(manager, "fetch_candidates", return_value=[]),
            mock.patch.object(manager, "stop_active_openvpn") as stop_mock,
            mock.patch.object(manager, "log_to_json"),
        ):
            result = manager.maintain_valid_nodes(force=True)

        self.assertEqual("没有拉取到新节点", result)
        self.assertTrue(process.running)
        stop_mock.assert_not_called()

    def test_fetch_timeout_skips_insecure_https_retry(self) -> None:
        csv_text = valid_snapshot()

        def fake_fetch(url, verify_ssl):
            if url.startswith("https://"):
                raise TimeoutError("timed out")
            return csv_text

        with (
            mock.patch.object(manager, "fetch_api_text", side_effect=fake_fetch) as fetch_mock,
            mock.patch.object(manager, "load_blacklist", return_value={}),
            mock.patch.object(manager, "set_state"),
            mock.patch.object(manager, "log_to_json"),
        ):
            nodes = manager.fetch_candidates()

        self.assertEqual(1, len(nodes))
        self.assertEqual(
            [mock.call(manager.API_HTTPS_URL, True), mock.call(manager.API_HTTP_URL, True)],
            fetch_mock.call_args_list,
        )

    def test_discovery_countries_are_normalized_and_persisted(self) -> None:
        countries = manager.persist_discovery_countries(["jp", "US", "JP", "bad", ""])

        self.assertEqual(["JP", "US"], countries)
        self.assertEqual(["JP", "US"], manager.load_ui_config()["discovery_countries"])
        self.assertEqual(["JP", "US"], manager.get_state()["discovery_countries"])

    def test_fetch_filters_country_after_source_is_accepted(self) -> None:
        csv_text = valid_snapshot_rows(
            [
                ("198.51.100.60", "Japan", "JP"),
                ("198.51.100.61", "United States", "US"),
            ]
        )
        manager.persist_discovery_countries(["JP"])

        with (
            mock.patch.object(manager, "fetch_api_text", return_value=csv_text) as fetch_mock,
            mock.patch.object(manager, "load_blacklist", return_value={}),
            mock.patch.object(manager, "log_to_json"),
        ):
            nodes = manager.fetch_candidates()

        self.assertEqual(["JP"], [node["country_short"] for node in nodes])
        fetch_mock.assert_called_once_with(manager.API_HTTPS_URL, True)
        self.assertEqual(csv_text, manager.API_CACHE_FILE.read_text(encoding="utf-8"))
        self.assertIn("成功获取 2 个", manager.get_state()["last_fetch_message"])
        self.assertIn("保留 1 个", manager.get_state()["last_fetch_message"])

    def test_empty_country_result_does_not_fall_through_to_next_source(self) -> None:
        csv_text = valid_snapshot_rows(
            [
                ("198.51.100.70", "Japan", "JP"),
                ("198.51.100.71", "United States", "US"),
            ]
        )
        manager.persist_discovery_countries(["DE"])

        with (
            mock.patch.object(manager, "fetch_api_text", return_value=csv_text) as fetch_mock,
            mock.patch.object(manager, "load_blacklist", return_value={}),
            mock.patch.object(manager, "log_to_json"),
        ):
            nodes = manager.fetch_candidates()

        self.assertEqual([], nodes)
        fetch_mock.assert_called_once_with(manager.API_HTTPS_URL, True)
        state = manager.get_state()
        self.assertEqual("ok", state["last_fetch_status"])
        self.assertEqual("official_https", state["last_fetch_source"])
        self.assertIn("保留 0 个", state["last_fetch_message"])

    def test_node_table_contains_latency_country_panel_and_test_action(self) -> None:
        self.assertIn('<th style="width: 125px;">延迟</th>', manager.INDEX_HTML)
        self.assertIn('colspan="7"', manager.INDEX_HTML)
        self.assertIn('class="country-option-input"', manager.INDEX_HTML)
        self.assertIn('${testBtn}', manager.INDEX_HTML)

    def test_web_dashboard_has_browser_freeze_safeguards(self) -> None:
        self.assertNotIn("backdrop-filter", manager.LOGIN_HTML)
        self.assertNotIn("backdrop-filter", manager.INDEX_HTML)
        self.assertNotIn("background-attachment: fixed", manager.INDEX_HTML)
        self.assertIn("@media (prefers-reduced-motion: reduce)", manager.LOGIN_HTML)
        self.assertIn("@media (prefers-reduced-motion: reduce)", manager.INDEX_HTML)
        self.assertIn("const pageSize = 50;", manager.INDEX_HTML)
        self.assertIn('id="pagination_container"', manager.INDEX_HTML)
        self.assertIn('paginationContainer.style.display = totalPages > 1 ? "flex" : "none";', manager.INDEX_HTML)
        self.assertIn("const MAX_RENDERED_LOG_LINES = 300;", manager.INDEX_HTML)
        self.assertIn("nodesRequestPromise", manager.INDEX_HTML)
        self.assertIn("backgroundPollInFlight", manager.INDEX_HTML)
        self.assertIn('let lastNodesSnapshotSignature = "";', manager.INDEX_HTML)
        self.assertIn("if (signature === lastNodesSnapshotSignature) return false;", manager.INDEX_HTML)
        self.assertIn('typeof document.hidden !== "boolean" || !document.hidden', manager.INDEX_HTML)
        self.assertEqual(500, manager.WEB_LOG_MAX_ENTRIES)

    def test_web_dashboard_has_cross_browser_interaction_safeguards(self) -> None:
        self.assertNotIn("fonts.googleapis.com", manager.LOGIN_HTML)
        self.assertNotIn("fonts.googleapis.com", manager.INDEX_HTML)
        self.assertIn('const pwd = document.getElementById("password").value;', manager.LOGIN_HTML)
        self.assertIn('const password = $("cred_password").value;', manager.INDEX_HTML)
        self.assertIn("function fetchWithTimeout", manager.LOGIN_HTML)
        self.assertIn("function fetchWithTimeout", manager.INDEX_HTML)
        self.assertNotIn("await fetch(", manager.INDEX_HTML)
        self.assertIn('role="dialog" aria-modal="true"', manager.INDEX_HTML)
        self.assertIn('aria-label="关闭网页安全设置"', manager.INDEX_HTML)
        self.assertIn('class="option-card active" data-value="auto" aria-pressed="true"', manager.INDEX_HTML)
        self.assertIn('class="vps-recommend-tab"', manager.INDEX_HTML)
        self.assertIn('position: static;', manager.INDEX_HTML)
        self.assertIn('-webkit-overflow-scrolling: touch;', manager.INDEX_HTML)
        self.assertIn('formatUrlHost(window.location.hostname)', manager.INDEX_HTML)
        self.assertNotIn('id="status" class="status" style="display: none;"', manager.INDEX_HTML)
        self.assertIn('${esc(localProxy)}', manager.INDEX_HTML)
        self.assertIn('${esc(statusMessage)}', manager.INDEX_HTML)

    def test_dashboard_javascript_is_valid(self) -> None:
        if not shutil.which("node"):
            self.skipTest("Node.js is not installed; JavaScript syntax check skipped")
        scripts = re.findall(r"<script>(.*?)</script>", manager.INDEX_HTML, re.DOTALL)
        self.assertTrue(scripts)
        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as handle:
            handle.write("\n".join(scripts))
            script_path = handle.name
        try:
            result = subprocess.run(
                ["node", "--check", script_path],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
        finally:
            Path(script_path).unlink(missing_ok=True)

    def test_random_password_uses_cryptographic_randomness(self) -> None:
        with mock.patch.object(manager.secrets, "choice", side_effect=list("aA0aA0aA0aA0")) as choice:
            password = manager.generate_random_password()

        self.assertEqual("aA0aA0aA0aA0", password)
        self.assertEqual(12, choice.call_count)

    def test_expired_sessions_are_removed(self) -> None:
        manager.active_sessions.update({"expired": 99.0, "active": 101.0})

        removed = manager.purge_expired_sessions(now=100.0)

        self.assertEqual(1, removed)
        self.assertEqual({"active": 101.0}, manager.active_sessions)

    def test_web_log_reader_only_returns_recent_valid_entries(self) -> None:
        log_file = manager.DATA_DIR / "logs" / "current.json"
        log_file.parent.mkdir(parents=True)
        with log_file.open("w", encoding="utf-8") as f:
            for index in range(520):
                f.write(json.dumps({"index": index}) + "\n")
            f.write("not-json\n")

        entries = manager.read_recent_log_entries(log_file)

        self.assertEqual(500, len(entries))
        self.assertEqual(20, entries[0]["index"])
        self.assertEqual(519, entries[-1]["index"])

    def test_web_update_controls_only_expose_stable_main_channel(self) -> None:
        self.assertEqual("2.1.5", manager.APP_VERSION)
        self.assertEqual("V2.1.5 正式版", manager.APP_VERSION_LABEL)
        self.assertIn("检测更新", manager.INDEX_HTML)
        self.assertIn("/api/check_update", manager.INDEX_HTML)
        self.assertIn("/tree/main", manager.INDEX_HTML)
        self.assertIn("/releases/latest", manager.INDEX_HTML)
        self.assertNotIn("/tree/bate", manager.INDEX_HTML)
        self.assertNotIn(">测试版<", manager.INDEX_HTML)
        self.assertIn('id="deployment_mode_label"', manager.INDEX_HTML)

    def test_installer_updates_only_from_main(self) -> None:
        install_text = (manager.ROOT_DIR / "install.sh").read_text(encoding="utf-8")

        self.assertIn('DEPLOY_BRANCH="main"', install_text)
        self.assertIn('branch = "main"', install_text)
        self.assertNotIn("CURRENT_BRANCH", install_text)
        self.assertNotIn("origin/master", install_text)
        self.assertNotIn("bate", install_text.lower())

    def test_installer_uses_secure_credentials_and_current_version(self) -> None:
        install_text = (manager.ROOT_DIR / "install.sh").read_text(encoding="utf-8")

        self.assertNotIn("random.choices", install_text)
        self.assertIn("secrets.choice", install_text)
        self.assertIn('get_app_version()', install_text)
        self.assertNotIn("管理终端 v2.0", install_text)
        self.assertIn("5-90 秒", install_text)
        self.assertIn('new_pwd = input("请输入新管理密码 (不能为空): ")', install_text)
        self.assertIn('state["active_openvpn_node_id"] = ""', install_text)
        self.assertIn("ip link show dev tun0", install_text)
        self.assertIn("pidof openvpn", install_text)
        self.assertIn('chmod 600 "$AUTH_FILE"', install_text)
        self.assertIn("FAMILY_IP_NONINTERACTIVE", install_text)
        self.assertIn('["ip", "rule", "del", "table", "100"]', install_text)
        self.assertIn('/etc/sysctl.d/99-family-ip.conf', install_text)
        self.assertNotIn('http://[::1]:${PROXY_PORT}', install_text)

    def test_openvpn_command_requires_server_certificate_usage(self) -> None:
        with mock.patch.object(manager, "get_openvpn_version", return_value=2.5):
            command = manager.openvpn_command("node.ovpn", route_nopull=True)
        index = command.index("--remote-cert-tls")
        self.assertEqual("server", command[index + 1])

    def test_release_workflow_uses_full_patch_version(self) -> None:
        workflow_text = (manager.ROOT_DIR / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

        self.assertIn("default: v2.1.5", workflow_text)
        self.assertIn("Family-IP V$(tr -d '\\r\\n' < VERSION) 正式版", workflow_text)
        self.assertNotIn("cut -d. -f1,2 VERSION", workflow_text)

    def test_latest_release_check_ignores_non_version_name_text(self) -> None:
        release = {
            "tag_name": "v2.2.0",
            "name": "Family-IP V2.2 正式版",
            "published_at": "2026-09-01T00:00:00Z",
            "draft": False,
            "prerelease": False,
        }
        with mock.patch.object(manager, "fetch_api_text", return_value=json.dumps(release)) as fetch_mock:
            result = manager.check_latest_release()

        self.assertTrue(result["ok"])
        self.assertTrue(result["update_available"])
        self.assertEqual("2.2.0", result["latest_version"])
        self.assertEqual("v2.2.0", result["latest_tag"])
        self.assertEqual(
            "https://github.com/horse-gemini/family-ip/releases/tag/v2.2.0",
            result["release_url"],
        )
        fetch_mock.assert_called_once_with(manager.GITHUB_LATEST_RELEASE_API, True)

    def test_latest_release_check_reports_current_formal_version(self) -> None:
        release = {
            "tag_name": "v2.1.5",
            "name": "Family-IP V2.1.5 正式版",
            "draft": False,
            "prerelease": False,
        }
        with mock.patch.object(manager, "fetch_api_text", return_value=json.dumps(release)):
            result = manager.check_latest_release()

        self.assertFalse(result["update_available"])
        self.assertEqual("V2.1.5 正式版", result["current_version_label"])

    def test_latest_release_check_reports_source_update_command(self) -> None:
        release = {"tag_name": "v2.2.0", "draft": False, "prerelease": False}
        with (
            mock.patch.object(manager, "fetch_api_text", return_value=json.dumps(release)),
            mock.patch.object(manager, "DEPLOYMENT_MODE", "source"),
            mock.patch.object(manager, "DEPLOYMENT_MODE_LABEL", "Python 源码"),
            mock.patch.object(manager, "UPDATE_COMMAND", "ml update"),
        ):
            result = manager.check_latest_release()

        self.assertEqual("source", result["deployment_mode"])
        self.assertEqual("ml update", result["update_command"])

    def test_latest_release_check_reports_docker_update_command(self) -> None:
        release = {"tag_name": "v2.2.0", "draft": False, "prerelease": False}
        with (
            mock.patch.object(manager, "fetch_api_text", return_value=json.dumps(release)),
            mock.patch.object(manager, "DEPLOYMENT_MODE", "docker"),
            mock.patch.object(manager, "DEPLOYMENT_MODE_LABEL", "Docker 容器"),
            mock.patch.object(
                manager,
                "UPDATE_COMMAND",
                "docker compose pull && docker compose up -d",
            ),
        ):
            result = manager.check_latest_release()

        self.assertEqual("docker", result["deployment_mode"])
        self.assertEqual(
            "docker compose pull && docker compose up -d",
            result["update_command"],
        )

    def test_fetch_uses_github_mirror_after_official_sources(self) -> None:
        csv_text = valid_snapshot()

        def fake_fetch(url, verify_ssl):
            if url == manager.MIRROR_HTTPS_URL:
                return csv_text
            raise TimeoutError("blocked")

        with (
            mock.patch.object(manager, "fetch_api_text", side_effect=fake_fetch) as fetch_mock,
            mock.patch.object(manager, "load_blacklist", return_value={}),
            mock.patch.object(manager, "log_to_json"),
            mock.patch.object(manager, "read_mirror_freshness", return_value=(0.0, "")),
        ):
            nodes = manager.fetch_candidates()

        self.assertEqual(1, len(nodes))
        self.assertEqual(
            [manager.API_HTTPS_URL, manager.API_HTTP_URL, manager.MIRROR_HTTPS_URL],
            [call.args[0] for call in fetch_mock.call_args_list],
        )
        self.assertEqual(csv_text, manager.API_CACHE_FILE.read_text(encoding="utf-8"))
        self.assertEqual("github_pages_https", manager.get_state()["last_fetch_source"])

    def test_http_source_does_not_replace_trusted_cache(self) -> None:
        cached_text = valid_snapshot("198.51.100.20")
        http_text = valid_snapshot("198.51.100.21")
        manager.API_CACHE_FILE.write_text(cached_text, encoding="utf-8")

        def fake_fetch(url, verify_ssl):
            if url == manager.API_HTTP_URL:
                return http_text
            raise TimeoutError("TLS unavailable")

        with (
            mock.patch.object(manager, "fetch_api_text", side_effect=fake_fetch),
            mock.patch.object(manager, "load_blacklist", return_value={}),
            mock.patch.object(manager, "log_to_json"),
        ):
            nodes = manager.fetch_candidates()

        self.assertEqual("198.51.100.21", nodes[0]["ip"])
        self.assertEqual(cached_text, manager.API_CACHE_FILE.read_text(encoding="utf-8"))

    def test_fetch_falls_back_to_local_cache(self) -> None:
        cached_text = valid_snapshot("198.51.100.30")
        manager.API_CACHE_FILE.write_text(cached_text, encoding="utf-8")

        with (
            mock.patch.object(manager, "fetch_api_text", side_effect=TimeoutError("all blocked")),
            mock.patch.object(manager, "load_blacklist", return_value={}),
            mock.patch.object(manager, "log_to_json"),
        ):
            nodes = manager.fetch_candidates()

        self.assertEqual("198.51.100.30", nodes[0]["ip"])
        self.assertEqual("local_cache", manager.get_state()["last_fetch_source"])

    def test_bundled_snapshot_seeds_local_cache(self) -> None:
        bundled_text = valid_snapshot("198.51.100.40")
        manager.BUNDLED_SNAPSHOT_FILE.write_text(bundled_text, encoding="utf-8")

        with (
            mock.patch.object(manager, "fetch_api_text", side_effect=TimeoutError("all blocked")),
            mock.patch.object(manager, "load_blacklist", return_value={}),
            mock.patch.object(manager, "log_to_json"),
        ):
            nodes = manager.fetch_candidates()

        self.assertEqual("198.51.100.40", nodes[0]["ip"])
        self.assertEqual(bundled_text, manager.API_CACHE_FILE.read_text(encoding="utf-8"))
        self.assertEqual("bundled_initial", manager.get_state()["last_fetch_source"])

    def test_snapshot_rejects_executable_openvpn_directive(self) -> None:
        unsafe_config = (
            "client\ndev tun\nproto udp\nremote 198.51.100.50 1194 udp\n"
            "script-security 2\nup /tmp/payload\n"
            "<ca>\nCA\n</ca>\n<cert>\nCERT\n</cert>\n<key>\nKEY\n</key>\n"
        )
        encoded = base64.b64encode(unsafe_config.encode("utf-8")).decode("ascii")
        csv_text = (
            "#HostName,IP,Score,Ping,Speed,CountryLong,CountryShort,NumVpnSessions,OpenVPN_ConfigData_Base64\n"
            f"vpn.example,198.51.100.50,100,20,1000,Japan,JP,1,{encoded}\n"
        )

        with self.assertRaisesRegex(ValueError, "no valid nodes"):
            snapshot_utils.parse_and_validate_snapshot(csv_text)


    # ---- stability: failover, health monitoring and self-healing ----

    def _available_pool(self) -> list[dict]:
        nodes = self.write_nodes(3)
        for node in nodes:
            node["probe_status"] = "available"
            node["latency_ms"] = 10
        manager.write_json(manager.NODES_FILE, nodes)
        return nodes

    def test_auto_switch_tries_next_candidate_after_a_failed_node(self) -> None:
        nodes = self._available_pool()
        attempts: list[str] = []

        def fake_connect(node_id: str) -> str:
            attempts.append(node_id)
            if len(attempts) == 1:
                stored = manager.read_nodes()
                for item in stored:
                    if item["id"] == node_id:
                        item["probe_status"] = "unavailable"
                manager.write_json(manager.NODES_FILE, stored)
                raise RuntimeError("handshake failed")
            return "connected"

        with (
            mock.patch.object(manager, "connect_node", fake_connect),
            mock.patch.object(manager, "log_to_json"),
        ):
            manager.auto_switch_node()

        self.assertEqual([nodes[0]["id"], nodes[1]["id"]], attempts)

    def test_auto_switch_yields_when_another_switch_is_running(self) -> None:
        self._available_pool()
        attempts: list[str] = []

        def busy_connect(node_id: str) -> str:
            attempts.append(node_id)
            raise manager.ConnectionBusy("当前已有连接切换任务正在运行，请稍后再试")

        with (
            mock.patch.object(manager, "connect_node", busy_connect),
            mock.patch.object(manager, "schedule_background_refill") as schedule_mock,
            mock.patch.object(manager, "log_to_json"),
        ):
            manager.auto_switch_node()

        # A busy switch is not a node failure: no retries, no refill, nodes stay available.
        self.assertEqual(1, len(attempts))
        schedule_mock.assert_not_called()
        self.assertTrue(all(n["probe_status"] == "available" for n in manager.read_nodes()))

    def test_auto_switch_without_candidates_does_not_tear_down_inflight_connection(self) -> None:
        self.write_nodes(1)
        self.assertTrue(manager.connection_attempt_lock.acquire(blocking=False))
        with (
            mock.patch.object(manager, "clear_active_connection_state") as clear_mock,
            mock.patch.object(manager, "test_multiple_nodes", return_value=[]),
            mock.patch.object(manager, "schedule_background_refill") as schedule_mock,
            mock.patch.object(manager, "log_to_json"),
        ):
            manager.auto_switch_node()

        clear_mock.assert_not_called()
        schedule_mock.assert_not_called()

    def test_connection_attempt_is_allowed_while_maintenance_is_probing(self) -> None:
        manager.maintenance_busy = True

        token, cancel_event = manager.begin_connection_attempt()
        try:
            self.assertTrue(manager.is_connecting)
            with self.assertRaises(manager.ConnectionBusy):
                manager.begin_connection_attempt()
        finally:
            manager.finish_connection_attempt(token, cancel_event)

        self.assertFalse(manager.is_connecting)
        # The UI still reports a busy state for the whole maintenance run.
        self.assertTrue(manager.get_state()["is_connecting"])

    def test_health_check_recovers_dead_tunnel_during_maintenance(self) -> None:
        nodes = self._available_pool()
        nodes[0]["active"] = True
        manager.write_json(manager.NODES_FILE, nodes)
        dead = FakeProcess()
        dead.running = False
        manager.active_openvpn_process = dead
        manager.active_openvpn_node_id = nodes[0]["id"]
        manager.maintenance_busy = True

        with (
            mock.patch.object(manager, "check_proxy_health") as health_mock,
            mock.patch.object(manager, "auto_switch_node") as switch_mock,
            mock.patch.object(manager, "log_to_json"),
        ):
            manager.run_proxy_health_check_once()

        # No slow exit probe for a process that is already gone, and recovery is immediate.
        health_mock.assert_not_called()
        switch_mock.assert_called_once_with()
        stored = {n["id"]: n for n in manager.read_nodes()}
        self.assertEqual("unavailable", stored[nodes[0]["id"]]["probe_status"])
        self.assertIn(nodes[0]["id"], manager.load_blacklist())

    def test_health_check_is_skipped_while_a_switch_is_in_progress(self) -> None:
        manager.is_connecting = True
        with mock.patch.object(manager, "check_proxy_health") as health_mock:
            manager.run_proxy_health_check_once()
        health_mock.assert_not_called()

    def test_health_check_does_not_report_failure_before_first_connection(self) -> None:
        manager.maintenance_busy = True
        with mock.patch.object(manager, "check_proxy_health") as health_mock:
            manager.run_proxy_health_check_once()
        health_mock.assert_not_called()

    def test_health_check_waits_for_failure_threshold_before_switching(self) -> None:
        nodes = self._available_pool()
        manager.active_openvpn_process = FakeProcess()
        manager.active_openvpn_node_id = nodes[0]["id"]

        with (
            mock.patch.object(manager, "check_proxy_health", return_value={"ok": False, "error": "timeout"}),
            mock.patch.object(manager, "repair_policy_routing_if_needed", return_value=False),
            mock.patch.object(manager, "PROXY_FAILURE_THRESHOLD", 3),
            mock.patch.object(manager, "auto_switch_node") as switch_mock,
            mock.patch.object(manager, "log_to_json"),
        ):
            manager.run_proxy_health_check_once()
            manager.run_proxy_health_check_once()
            switch_mock.assert_not_called()
            manager.run_proxy_health_check_once()

        switch_mock.assert_called_once_with()

    def test_health_check_repairs_lost_policy_route_instead_of_switching(self) -> None:
        nodes = self._available_pool()
        manager.active_openvpn_process = FakeProcess()
        manager.active_openvpn_node_id = nodes[0]["id"]
        results = [
            {"ok": False, "error": "no route"},
            {"ok": True, "ip": "203.0.113.7", "latency_ms": 120},
        ]

        with (
            mock.patch.object(manager, "check_proxy_health", side_effect=results) as health_mock,
            mock.patch.object(manager, "repair_policy_routing_if_needed", return_value=True) as repair_mock,
            mock.patch.object(manager, "auto_switch_node") as switch_mock,
            mock.patch.object(manager, "log_to_json"),
        ):
            manager.run_proxy_health_check_once()

        repair_mock.assert_called_once_with("tun0")
        self.assertEqual(2, health_mock.call_count)
        switch_mock.assert_not_called()
        self.assertEqual(0, manager.consecutive_proxy_failures)
        state = manager.get_state()
        self.assertTrue(state["proxy_ok"])
        self.assertEqual("203.0.113.7", state["proxy_ip"])

    def test_policy_routing_intact_detects_missing_route_or_rule(self) -> None:
        def fake_ip(route_out: str, rule_out: str, route_rc: int = 0):
            def run(cmd, **kwargs):
                if cmd[:3] == ["ip", "route", "show"]:
                    return subprocess.CompletedProcess(cmd, route_rc, route_out, "")
                return subprocess.CompletedProcess(cmd, 0, rule_out, "")
            return run

        rules = "0:\tfrom all lookup local\n32765:\tfrom all oif tun0 lookup 100\n32766:\tfrom all lookup main\n"
        with mock.patch.object(manager.sys, "platform", "linux"):
            with mock.patch.object(manager.subprocess, "run", fake_ip("default dev tun0 scope link\n", rules)):
                self.assertTrue(manager.policy_routing_intact("tun0"))
            with mock.patch.object(manager.subprocess, "run", fake_ip("", rules)):
                self.assertFalse(manager.policy_routing_intact("tun0"))
            with mock.patch.object(manager.subprocess, "run", fake_ip("", rules, route_rc=2)):
                self.assertFalse(manager.policy_routing_intact("tun0"))
            with mock.patch.object(
                manager.subprocess, "run",
                fake_ip("default dev tun0 scope link\n", "0:\tfrom all lookup local\n"),
            ):
                self.assertFalse(manager.policy_routing_intact("tun0"))
            with mock.patch.object(manager.subprocess, "run", side_effect=FileNotFoundError("ip")):
                self.assertIsNone(manager.policy_routing_intact("tun0"))

    def test_repair_policy_routing_only_when_tunnel_is_up_and_route_is_missing(self) -> None:
        with mock.patch.object(manager, "setup_policy_routing", return_value=True) as setup_mock:
            # No running tunnel: nothing to repair.
            self.assertFalse(manager.repair_policy_routing_if_needed("tun0"))
            manager.active_openvpn_process = FakeProcess()
            manager.active_openvpn_node_id = "node-0"
            with (
                mock.patch.object(manager.Path, "exists", return_value=True),
                mock.patch.object(manager, "log_to_json"),
            ):
                with mock.patch.object(manager, "policy_routing_intact", return_value=True):
                    self.assertFalse(manager.repair_policy_routing_if_needed("tun0"))
                with mock.patch.object(manager, "policy_routing_intact", return_value=None):
                    self.assertFalse(manager.repair_policy_routing_if_needed("tun0"))
                setup_mock.assert_not_called()
                with mock.patch.object(manager, "policy_routing_intact", return_value=False):
                    self.assertTrue(manager.repair_policy_routing_if_needed("tun0"))
        setup_mock.assert_called_once_with("tun0")

    def test_setup_policy_routing_is_idempotent_and_clears_duplicate_rules(self) -> None:
        commands: list[list[str]] = []
        remaining_rules = [3]

        def fake_run(cmd, **kwargs):
            commands.append(list(cmd))
            if cmd[:3] == ["ip", "rule", "del"]:
                if remaining_rules[0] > 0:
                    remaining_rules[0] -= 1
                    return subprocess.CompletedProcess(cmd, 0, b"", b"")
                return subprocess.CompletedProcess(cmd, 2, b"", b"no such rule")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with mock.patch.object(manager.subprocess, "run", fake_run):
            self.assertTrue(manager.setup_policy_routing("tun0"))

        self.assertEqual(4, sum(cmd[:3] == ["ip", "rule", "del"] for cmd in commands))
        self.assertIn(["ip", "route", "replace", "default", "dev", "tun0", "table", "100"], commands)
        self.assertNotIn(["ip", "route", "add", "default", "dev", "tun0", "table", "100"], commands)
        self.assertEqual(1, sum(cmd[:3] == ["ip", "rule", "add"] for cmd in commands))

    def test_proxy_check_wakes_early_once_when_tunnel_process_exits(self) -> None:
        dead = FakeProcess()
        dead.running = False
        manager.active_openvpn_process = dead
        manager.active_openvpn_node_id = "node-0"
        manager._exit_wake_process = None

        started = manager.time.time()
        manager.wait_for_next_proxy_check(30)
        self.assertLess(manager.time.time() - started, 5)

        # The same dead process must not cause a busy loop of early wake-ups.
        started = manager.time.time()
        manager.wait_for_next_proxy_check(0.3)
        self.assertGreaterEqual(manager.time.time() - started, 0.25)

    def test_no_tunnel_retry_backs_off_instead_of_waiting_a_full_cycle(self) -> None:
        with (
            mock.patch.object(manager, "CHECK_INTERVAL_SECONDS", 1260),
            mock.patch.object(manager, "NO_TUNNEL_RETRY_MAX_SECONDS", 300),
        ):
            delays = [manager.no_tunnel_retry_delay(n) for n in (1, 2, 3, 4, 5, 50)]
        self.assertEqual([30, 60, 120, 240, 300, 300], delays)
        with (
            mock.patch.object(manager, "CHECK_INTERVAL_SECONDS", 45),
            mock.patch.object(manager, "NO_TUNNEL_RETRY_MAX_SECONDS", 300),
        ):
            self.assertEqual(45, manager.no_tunnel_retry_delay(9))

    def test_exit_ip_check_requires_a_real_ip_address(self) -> None:
        self.assertTrue(manager.looks_like_ip_address("203.0.113.9"))
        self.assertTrue(manager.looks_like_ip_address(" 2001:db8::1 "))
        self.assertFalse(manager.looks_like_ip_address("<html>blocked</html>"))
        self.assertFalse(manager.looks_like_ip_address(""))

    def _run_health_check_with_curl_outputs(self, outputs: list[str]) -> tuple[dict, int]:
        calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            index = min(calls["n"], len(outputs) - 1)
            calls["n"] += 1
            return subprocess.CompletedProcess(cmd, 0, outputs[index], "")

        listener = mock.MagicMock()
        with (
            mock.patch.object(manager.socket, "socket", return_value=listener),
            mock.patch.object(manager.Path, "exists", return_value=True),
            mock.patch.object(manager.subprocess, "run", fake_run),
            mock.patch.object(manager.proxy_server, "get_proxy_credentials", return_value=(None, None)),
            mock.patch.object(manager.vpn_utils, "diagnose_local_obstructions", return_value=None),
        ):
            return manager.check_proxy_health(), calls["n"]

    def test_exit_ip_check_accepts_trailing_newline_from_ip_service(self) -> None:
        result, calls = self._run_health_check_with_curl_outputs(["203.0.113.9\n\n0.412 200"])
        self.assertEqual({"ok": True, "ip": "203.0.113.9", "latency_ms": 412}, result)
        self.assertEqual(1, calls)

    def test_exit_ip_check_falls_through_hijacked_page_to_next_service(self) -> None:
        result, calls = self._run_health_check_with_curl_outputs(
            ["<html>captive portal</html>\n0.050 200", "198.51.100.4\n0.300 200"]
        )
        self.assertTrue(result["ok"])
        self.assertEqual("198.51.100.4", result["ip"])
        self.assertEqual(2, calls)

    def test_exit_ip_check_fails_when_no_service_returns_an_ip(self) -> None:
        result, calls = self._run_health_check_with_curl_outputs(["<html>blocked</html>\n0.050 200"])
        self.assertFalse(result["ok"])
        self.assertEqual(len(manager.PROXY_HEALTH_CHECK_URLS), calls)

    def test_node_refresh_keeps_ip_intel_and_pending_connection_target(self) -> None:
        existing = self.write_nodes(3)
        existing[0].update(
            country_short="US", ip_type="residential", ip_type_confidence="medium",
            geo_country_short="US", is_vpn=False, abuser_score=0.01, risk_control_index=10,
            probe_status="available", latency_ms=77,
        )
        manager.write_json(manager.NODES_FILE, existing)
        # node-2 is being connected right now but has dropped out of the fresh list.
        manager.pending_connection_node_id = existing[2]["id"]
        fresh = [
            {**node, "probe_status": "not_checked", "latency_ms": 0}
            for node in self.write_nodes(2)
        ]
        fresh[0]["country_short"] = "US"
        manager.write_json(manager.NODES_FILE, existing)
        manager.active_openvpn_process = FakeProcess()
        manager.active_openvpn_node_id = "missing-from-list"

        with (
            mock.patch.object(manager, "fetch_candidates", return_value=fresh),
            mock.patch.object(manager, "test_multiple_nodes", return_value=[]),
            mock.patch.object(manager, "maybe_switch_to_us_node"),
            mock.patch.object(manager, "log_to_json"),
        ):
            manager.maintain_valid_nodes()

        stored = {node["id"]: node for node in manager.read_nodes()}
        self.assertIn(existing[2]["id"], stored)
        refreshed = stored[existing[0]["id"]]
        self.assertEqual("medium", refreshed["ip_type_confidence"])
        self.assertEqual("US", refreshed["geo_country_short"])
        self.assertEqual("available", refreshed["probe_status"])
        self.assertTrue(manager.is_us_residential_node(refreshed))
        self.assertFalse(manager.maintenance_busy)

    def test_source_outage_reconnects_from_existing_available_pool(self) -> None:
        self._available_pool()

        with (
            mock.patch.object(manager, "fetch_candidates", side_effect=RuntimeError("[ERR_API] all sources down")),
            mock.patch.object(manager.vpn_utils, "check_and_fix_dns"),
            mock.patch.object(manager, "auto_switch_node") as switch_mock,
            mock.patch.object(manager, "log_to_json"),
        ):
            result = manager.maintain_valid_nodes()

        self.assertEqual("没有拉取到新节点", result)
        switch_mock.assert_called_once_with()

    def test_failed_connect_does_not_overwrite_concurrent_node_updates(self) -> None:
        nodes = self.write_nodes(2)

        def fake_openvpn(*args, **kwargs):
            # A probe / enrichment thread updates the list while the handshake is running.
            stored = manager.read_nodes()
            stored[1]["probe_status"] = "available"
            stored[1]["ip_type"] = "residential"
            manager.write_json(manager.NODES_FILE, stored)
            return False, "handshake failed", None

        with (
            mock.patch.object(manager, "run_openvpn_until_ready", side_effect=fake_openvpn),
            mock.patch.object(manager, "cleanup_policy_routing"),
            mock.patch.object(manager, "delete_tun_device"),
            mock.patch.object(manager, "log_to_json"),
        ):
            with self.assertRaisesRegex(RuntimeError, "handshake failed"):
                manager.connect_node(nodes[0]["id"])

        stored = {node["id"]: node for node in manager.read_nodes()}
        self.assertEqual("unavailable", stored[nodes[0]["id"]]["probe_status"])
        self.assertEqual("available", stored[nodes[1]["id"]]["probe_status"])
        self.assertEqual("residential", stored[nodes[1]["id"]]["ip_type"])
        self.assertEqual("", manager.pending_connection_node_id)

    def test_us_enrichment_before_selection_only_targets_untyped_us_nodes(self) -> None:
        nodes = self.write_nodes(4)
        nodes[0].update(country_short="US")
        nodes[1].update(country_short="US", ip_type="hosting", ip_type_confidence="high")
        nodes[2].update(country_short="JP")
        nodes[3].update(country_short="US")
        manager.write_json(manager.NODES_FILE, nodes)
        seen: list[list[str]] = []

        def fake_enrich(batch: list[dict]) -> None:
            seen.append(sorted(node["id"] for node in batch))
            for node in batch:
                node.update(ip_type="residential", ip_type_confidence="medium", geo_country_short="US")

        with (
            mock.patch.object(manager.vpn_utils, "enrich_ip_info", fake_enrich),
            mock.patch.object(manager, "log_to_json"),
        ):
            manager.enrich_us_nodes_before_selection()

        self.assertEqual([[nodes[0]["id"], nodes[3]["id"]]], seen)
        stored = {node["id"]: node for node in manager.read_nodes()}
        self.assertTrue(manager.is_us_residential_node(stored[nodes[0]["id"]]))
        self.assertEqual("hosting", stored[nodes[1]["id"]]["ip_type"])
        self.assertFalse(stored[nodes[2]["id"]].get("ip_type"))

    def test_us_enrichment_before_selection_does_not_block_past_deadline(self) -> None:
        nodes = self.write_nodes(1)
        nodes[0].update(country_short="US")
        manager.write_json(manager.NODES_FILE, nodes)
        release = threading.Event()

        def slow_enrich(batch: list[dict]) -> None:
            release.wait(5)

        try:
            with (
                mock.patch.object(manager.vpn_utils, "enrich_ip_info", slow_enrich),
                mock.patch.object(manager, "US_ENRICH_DEADLINE_SECONDS", 0.2),
                mock.patch.object(manager, "log_to_json"),
            ):
                started = manager.time.time()
                manager.enrich_us_nodes_before_selection()
                elapsed = manager.time.time() - started
        finally:
            release.set()

        self.assertLess(elapsed, 2)

    def test_ip_cache_save_is_atomic_and_drops_expired_entries(self) -> None:
        now = manager.time.time()
        manager.vpn_utils.save_ip_cache({
            "198.51.100.1": {"ip_type": "residential", "cached_at": now},
            "198.51.100.2": {"ip_type": "hosting", "cached_at": now - manager.vpn_utils.IP_CACHE_TTL_SECONDS - 10},
            "198.51.100.3": "corrupted",
        })

        cache = manager.vpn_utils.load_ip_cache()
        self.assertEqual(["198.51.100.1"], list(cache))
        self.assertFalse(manager.vpn_utils.IP_CACHE_FILE.with_suffix(".json.tmp").exists())
        manager.vpn_utils.IP_CACHE_FILE.write_text("[1, 2, 3]", encoding="utf-8")
        self.assertEqual({}, manager.vpn_utils.load_ip_cache())

    def test_stop_process_reaps_a_process_that_ignores_sigterm(self) -> None:
        events: list[str] = []

        class StubbornProcess:
            def poll(self):
                return None

            def terminate(self) -> None:
                events.append("terminate")

            def wait(self, timeout=None):
                events.append(f"wait:{timeout}")
                if "kill" not in events:
                    raise subprocess.TimeoutExpired("openvpn", timeout)
                return -9

            def kill(self) -> None:
                events.append("kill")

        manager.stop_process(StubbornProcess())

        self.assertEqual(["terminate", "wait:8", "kill", "wait:3"], events)


    # ---- US node pool: keep scarce US nodes across list rotations ----

    def _fetch_with_sources(self, official_rows, mirror_rows=None, mirror_error=None):
        def fake_fetch(url, verify_ssl=True, **kwargs):
            if url == manager.MIRROR_HTTPS_URL:
                if mirror_error:
                    raise mirror_error
                return valid_snapshot_rows(mirror_rows or [("198.51.100.250", "Japan", "JP")])
            return valid_snapshot_rows(official_rows)

        with (
            mock.patch.object(manager, "fetch_api_text_with_deadline", fake_fetch),
            mock.patch.object(manager, "cache_api_snapshot"),
            mock.patch.object(manager, "log_to_json"),
        ):
            return manager.fetch_candidates()

    def test_us_node_is_retained_after_it_rotates_out_of_the_source(self) -> None:
        first = self._fetch_with_sources(
            [("198.51.100.10", "Japan", "JP"), ("203.0.113.50", "United States", "US")]
        )
        self.assertEqual({"198.51.100.10", "203.0.113.50"}, {n["ip"] for n in first})

        # The next list no longer contains the US node: it must still be a candidate.
        second = self._fetch_with_sources([("198.51.100.11", "Japan", "JP")])
        self.assertEqual({"198.51.100.11", "203.0.113.50"}, {n["ip"] for n in second})
        retained = next(n for n in second if n["ip"] == "203.0.113.50")
        self.assertEqual("US", retained["country_short"])
        self.assertTrue(manager.is_us_node(retained))
        # No duplicates when the node shows up in the source again.
        third = self._fetch_with_sources(
            [("203.0.113.50", "United States", "US"), ("198.51.100.11", "Japan", "JP")]
        )
        self.assertEqual(1, sum(n["ip"] == "203.0.113.50" for n in third))

    def test_us_pool_drops_vanished_nodes_but_keeps_working_ones(self) -> None:
        self._fetch_with_sources(
            [("203.0.113.50", "United States", "US"), ("203.0.113.51", "United States", "US"),
             ("198.51.100.10", "Japan", "JP")]
        )
        expired = manager.time.time() - manager.US_NODE_RETENTION_HOURS * 3600 - 60
        pool = manager.read_json(manager.US_NODE_POOL_FILE, {})
        self.assertEqual({"203.0.113.50", "203.0.113.51"}, set(pool))
        for entry in pool.values():
            entry["last_seen"] = expired
        manager.write_json(manager.US_NODE_POOL_FILE, pool)
        # .51 has gone from the source long ago but is still in the node list and still works.
        manager.write_json(manager.NODES_FILE, [
            {"id": "us-working", "ip": "203.0.113.51", "country_short": "US",
             "probe_status": "available", "probed_at": manager.time.time() - 30},
        ])
        # Simulate the retention bookkeeping having seen the working node before expiry.
        pool["203.0.113.51"]["last_ok"] = manager.time.time() - 30
        manager.write_json(manager.US_NODE_POOL_FILE, pool)

        candidates = self._fetch_with_sources([("198.51.100.11", "Japan", "JP")])

        self.assertEqual({"198.51.100.11", "203.0.113.51"}, {n["ip"] for n in candidates})
        self.assertEqual({"203.0.113.51"}, set(manager.load_us_node_pool()))

    def test_us_pool_records_successful_probes_for_retained_nodes(self) -> None:
        self._fetch_with_sources([("203.0.113.50", "United States", "US")])
        probed_at = manager.time.time() - 5
        manager.write_json(manager.NODES_FILE, [
            {"id": "us", "ip": "203.0.113.50", "country_short": "US",
             "probe_status": "available", "probed_at": probed_at},
        ])

        self._fetch_with_sources([("198.51.100.11", "Japan", "JP")])

        self.assertAlmostEqual(probed_at, manager.load_us_node_pool()["203.0.113.50"]["last_ok"], places=3)

    def test_stale_local_snapshot_does_not_refresh_us_pool(self) -> None:
        manager.API_CACHE_FILE.write_text(
            valid_snapshot_rows([("203.0.113.50", "United States", "US")]), encoding="utf-8"
        )
        with (
            mock.patch.object(manager, "fetch_api_text_with_deadline", side_effect=OSError("offline")),
            mock.patch.object(manager.vpn_utils, "diagnose_api_failure", return_value=(1001, "offline")),
            mock.patch.object(manager, "log_to_json"),
        ):
            candidates = manager.fetch_candidates()

        self.assertEqual(["203.0.113.50"], [n["ip"] for n in candidates])
        self.assertEqual({}, manager.load_us_node_pool())

    def test_us_pool_harvests_us_nodes_from_mirror(self) -> None:
        manager.last_us_mirror_harvest = 0.0
        candidates = self._fetch_with_sources(
            [("198.51.100.10", "Japan", "JP")],
            mirror_rows=[("203.0.113.77", "United States", "US"), ("198.51.100.99", "Japan", "JP")],
        )

        # Only the US node is taken from the mirror; the official list stays authoritative otherwise.
        self.assertEqual({"198.51.100.10", "203.0.113.77"}, {n["ip"] for n in candidates})
        self.assertEqual("github_pages_https", manager.load_us_node_pool()["203.0.113.77"]["source"])

    def test_us_pool_mirror_harvest_is_throttled_and_failure_is_harmless(self) -> None:
        manager.last_us_mirror_harvest = 0.0
        calls: list[str] = []

        def fake_fetch(url, verify_ssl=True, **kwargs):
            calls.append(url)
            if url == manager.MIRROR_HTTPS_URL:
                raise TimeoutError("mirror down")
            return valid_snapshot_rows([("198.51.100.10", "Japan", "JP")])

        with (
            mock.patch.object(manager, "fetch_api_text_with_deadline", fake_fetch),
            mock.patch.object(manager, "cache_api_snapshot"),
            mock.patch.object(manager, "log_to_json"),
        ):
            first = manager.fetch_candidates()
            second = manager.fetch_candidates()

        self.assertEqual(["198.51.100.10"], [n["ip"] for n in first])
        self.assertEqual(["198.51.100.10"], [n["ip"] for n in second])
        self.assertEqual(1, calls.count(manager.MIRROR_HTTPS_URL))

    def test_us_pool_cannot_inject_unsafe_openvpn_config(self) -> None:
        unsafe = base64.b64encode(
            b"client\ndev tun\nproto udp\nremote 203.0.113.66 1194 udp\nup /tmp/evil.sh\n"
            b"<ca>\nCA\n</ca>\n<cert>\nCERT\n</cert>\n<key>\nKEY\n</key>\n"
        ).decode("ascii")
        manager.write_json(manager.US_NODE_POOL_FILE, {
            "203.0.113.66": {
                "row": {"IP": "203.0.113.66", "CountryShort": "US", "CountryLong": "United States",
                        "OpenVPN_ConfigData_Base64": unsafe},
                "last_seen": manager.time.time(),
            },
            "203.0.113.67": {"row": "not-a-dict", "last_seen": manager.time.time()},
            "203.0.113.68": {
                "row": {"IP": "203.0.113.68", "CountryShort": "JP", "OpenVPN_ConfigData_Base64": unsafe},
                "last_seen": manager.time.time(),
            },
        })

        candidates = self._fetch_with_sources([("198.51.100.10", "Japan", "JP")])

        self.assertEqual(["198.51.100.10"], [n["ip"] for n in candidates])

    def test_us_pool_is_capped_and_can_be_disabled(self) -> None:
        rows = [(f"203.0.113.{i}", "United States", "US") for i in range(1, 7)]
        with mock.patch.object(manager, "US_NODE_POOL_MAX", 4):
            self._fetch_with_sources(rows)
            self.assertEqual(4, len(manager.load_us_node_pool()))
        with mock.patch.object(manager, "US_NODE_RETENTION_HOURS", 0):
            candidates = self._fetch_with_sources([("198.51.100.10", "Japan", "JP")])
        self.assertEqual(["198.51.100.10"], [n["ip"] for n in candidates])


class ProxyServerConcurrencyTests(unittest.TestCase):
    def test_socks5_rejects_client_without_no_auth_method(self) -> None:
        class Client:
            def __init__(self):
                self.incoming = bytearray(b"\x01\x02")
                self.sent = bytearray()
                self.closed = False

            def recv(self, size):
                chunk = self.incoming[:size]
                del self.incoming[:size]
                return bytes(chunk)

            def sendall(self, data):
                self.sent.extend(data)

            def close(self):
                self.closed = True

        client = Client()
        with mock.patch.object(proxy_server, "proxy_auth_enabled", return_value=False):
            proxy_server.socks5_client(client, b"\x05")

        self.assertEqual(b"\x05\xff", bytes(client.sent))
        self.assertTrue(client.closed)

    def test_each_proxy_worker_keeps_its_accepted_socket(self) -> None:
        class Client:
            def __init__(self, name):
                self.name = name

            def close(self):
                pass

        class FakeServer:
            def __init__(self):
                self.items = [(Client("first"), ("first", 1)), (Client("second"), ("second", 2))]

            def setsockopt(self, *args):
                pass

            def bind(self, *args):
                pass

            def listen(self, *args):
                pass

            def accept(self):
                if self.items:
                    return self.items.pop(0)
                raise KeyboardInterrupt()

        class DeferredThread:
            targets = []

            def __init__(self, target, daemon=True):
                self.target = target
                self.targets.append(target)

            def start(self):
                pass

        seen = []
        semaphore = mock.Mock()
        semaphore.acquire.return_value = True
        with (
            mock.patch.object(proxy_server.socket, "socket", return_value=FakeServer()),
            mock.patch.object(proxy_server.threading, "Thread", DeferredThread),
            mock.patch.object(
                proxy_server,
                "proxy_client",
                side_effect=lambda client, address: seen.append((client.name, address[0])),
            ),
            mock.patch.object(proxy_server, "proxy_connection_sem", semaphore),
        ):
            with self.assertRaises(KeyboardInterrupt):
                proxy_server.start_proxy_server("127.0.0.1", 7928)
            for target in DeferredThread.targets:
                target()

        self.assertEqual([("first", "first"), ("second", "second")], seen)


if __name__ == "__main__":
    unittest.main()
