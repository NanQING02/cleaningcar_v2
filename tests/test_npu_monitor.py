import unittest

from cleaningcar.npu_monitor import (
    format_npu_status,
    npu_status_flags,
    parse_devfreq_load,
    parse_rknpu_core_loads,
)


class NpuMonitorTest(unittest.TestCase):
    def test_parse_rknpu_core_loads(self):
        loads = parse_rknpu_core_loads("NPU load:  Core0:  3%, Core1:  45%, Core2: 100%,")
        self.assertEqual(loads, {0: 3.0, 1: 45.0, 2: 100.0})

    def test_parse_devfreq_load(self):
        load, freq = parse_devfreq_load("100@1000000000Hz")
        self.assertEqual(load, 100.0)
        self.assertEqual(freq, 1000000000)

    def test_format_and_flag_saturated_npu(self):
        status = {
            "available": True,
            "core_loads": {0: 10.0, 1: 95.0, 2: 20.0},
            "devfreq_load": 80.0,
            "cur_freq_mhz": 1000.0,
            "max_freq_mhz": 1000.0,
            "volt_v": 0.825,
            "governor": "rknpu_ondemand",
            "power": "on",
            "reset": "off",
        }
        self.assertEqual(npu_status_flags(status), ["npu_saturated"])
        text = format_npu_status(status)
        self.assertIn("c1:95%", text)
        self.assertIn("freq=1000/1000MHz", text)
        self.assertIn("flags=npu_saturated", text)


if __name__ == "__main__":
    unittest.main()
