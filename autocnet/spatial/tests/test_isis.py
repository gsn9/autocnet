"""This module has tests for the isis functions."""

# This is free and unencumbered software released into the public domain.
#
# The authors of autocnet do not claim copyright on the contents of this file.
# For more details about the LICENSE terms and the AUTHORS, you will
# find files of those names at the top level of this repository.
#
# SPDX-License-Identifier: CC0-1.0

import contextlib
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt


try:
    import kalasiris as isis
except Exception as exception:
    from autocnet.utils.utils import FailedImport
    isis = FailedImport(exception)

from autocnet.spatial import isis as si

class TestErrors(unittest.TestCase):

    def test_point_info(self):
        self.assertRaises(
            ValueError, si.point_info, "dummy.cub", 10, 10, "bogus"
        )
        self.assertRaises(
            IndexError,
            si.point_info,
            "dummy.cub",
            [10, 20, 30],
            [10, 20],
            "image"
        )
        self.assertRaises(
            TypeError, si.point_info, "dummy.cub", {10, 20}, {10, 20}, "image"
        )
        self.assertRaises(
            IndexError,
            si.point_info,
            "dummy.cub",
            np.array([[1, 2], [3, 4]]),
            np.array([1, 2, 3, 4]),
            "image"
        )
        self.assertRaises(
            IndexError,
            si.point_info,
            "dummy.cub",
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4]),
            "image"
        )

class TestISIS(unittest.TestCase):

    def setUp(self) -> None:
        self.resourcedir = Path("test-resources")
        self.red50img = self.resourcedir / "PSP_010502_2090_RED5_0.img"
        self.red51img = self.resourcedir / "PSP_010502_2090_RED5_1.img"

        if not all((self.red50img.exists(), self.red51img.exists())):
            self.skipTest(
                f"One or more files is missing from the "
                f"{self.resourcedir.resolve()} directory. "
                f"Tests on real files skipped."
            )

        self.cube = self.red50img.with_suffix(".TestISIS.cub")
        isis.hi2isis(self.red50img, to=self.cube)
        isis.spiceinit(self.cube)

        self.map = self.cube.with_suffix(".map.cub")
        isis.cam2map(self.cube, to=self.map)

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            self.cube.unlink()
            self.map.unlink()
            Path("print.prt").unlink()

    def test_point_info(self):
        self.assertRaises(
            ValueError, si.point_info, self.cube, -10, -10, "image"
        )

        d = si.point_info(self.cube, 10, 10, "image")
        self.assertEqual(10, d["Sample"])
        self.assertEqual(10, d["Line"])
        self.assertEqual(28.537311831691, d["PlanetocentricLatitude"].value)
        self.assertEqual(274.14960455269, d["PositiveEast360Longitude"].value)

        x_sample = 20
        x_lon = 274.14948072713
        y_line = 20
        y_lat = 28.537396673529

        d2 = si.point_info(self.cube, [10, x_sample], [10, y_line], "image")
        self.assertEqual(x_sample, d2[1]["Sample"])
        self.assertEqual(y_line, d2[1]["Line"])
        self.assertEqual(y_lat, d2[1]["PlanetocentricLatitude"].value)
        self.assertEqual(x_lon, d2[1]["PositiveEast360Longitude"].value)

        self.assertEqual(d, d2[0])

        d3 = si.point_info(self.cube, x_lon, y_lat, "ground")
        self.assertEqual(20.001087366213, d3["Sample"])
        self.assertEqual(20.004109124452, d3["Line"])

        d4 = si.point_info(self.map, 10, 10, "image")
        d5 = si.point_info(self.map, [10, x_sample], [10, y_line], "image")
        self.assertEqual(d4, d5[0])

        d6 = si.point_info(self.map, x_lon, y_lat, "ground")
        self.assertEqual(961.20490394075, d6["Sample"])
        self.assertEqual(3959.4515093358, d6["Line"])

    def test_image_to_ground(self):
        lon, lat = si.image_to_ground(self.cube, 20, 20)
        x = 274.14948
        y = 28.537396
        self.assertAlmostEqual(x, lon, places=5)
        self.assertAlmostEqual(y, lat, places=5)

        lons, lats = si.image_to_ground(
            self.cube, np.array([10, 20]), np.array([10, 20])
        )
        npt.assert_allclose(np.array([274.14961, x]), lons, rtol=1e-05)
        npt.assert_allclose(np.array([28.53731, y]), lats, rtol=1e-05)

        lon, lat = si.image_to_ground(self.map, 20, 20)
        x = 274.13914
        y = 28.57541
        self.assertAlmostEqual(x, lon, places=5)
        self.assertAlmostEqual(y, lat, places=5)

        lons, lats = si.image_to_ground(
            self.map, np.array([10, 20]), np.array([10, 20])
        )
        npt.assert_allclose(np.array([274.13903, x]), lons, rtol=1e-05)
        npt.assert_allclose(np.array([28.57551, y]), lats, rtol=1e-05)

    def test_ground_to_image(self):
        lon = 274.14948072713
        lat = 28.537396673529
        goal_samp = 20.001087
        goal_line = 20.004109
        sample, line = si.ground_to_image(
            self.cube, lon, lat
        )
        self.assertAlmostEqual(goal_samp, sample, places=6)
        self.assertAlmostEqual(goal_line, line, places=6)

        samples, lines = si.ground_to_image(
            self.cube,
            np.array([lon, 274.1495]),
            np.array([lat, 28.5374])
        )
        npt.assert_allclose(np.array([goal_samp, 18.241668]), samples)
        npt.assert_allclose(np.array([goal_line, 20.145382]), lines)

        lon = 274.13903475
        lat = 28.57550764
        goal_samp = 10.5001324
        goal_line = 10.49999466
        sample, line = si.ground_to_image(
            self.map, lon, lat
        )
        self.assertAlmostEqual(goal_samp, sample)
        self.assertAlmostEqual(goal_line, line)

        samples, lines = si.ground_to_image(
            self.map,
            np.array([lon, 274.14948072713]),
            np.array([lat, 28.57541113])
        )
        npt.assert_allclose(np.array([goal_samp, 961.03569217]), samples)
        npt.assert_allclose(np.array([goal_line, 20.50009032]), lines)
