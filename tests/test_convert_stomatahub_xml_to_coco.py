import tempfile
import unittest
from pathlib import Path

from PIL import Image

from convert_stomatahub_xml_to_coco import convert


class StomataHubConversionTests(unittest.TestCase):
    def test_conversion_clips_edges_and_rejects_degenerate_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            Image.new("RGB", (100, 80)).save(source / "sample.jpg")
            (source / "sample.xml").write_text(
                """
                <annotation>
                  <filename>sample.jpg</filename>
                  <size><width>100</width><height>80</height></size>
                  <object><class>stomata</class><bndbox>
                    <xmin>-2</xmin><ymin>10</ymin>
                    <xmax>20</xmax><ymax>30</ymax>
                  </bndbox></object>
                  <object><class>stomata</class><bndbox>
                    <xmin>50</xmin><ymin>50</ymin>
                    <xmax>50</xmax><ymax>70</ymax>
                  </bndbox></object>
                  <object><class>pore</class><bndbox>
                    <xmin>30</xmin><ymin>20</ymin>
                    <xmax>35</xmax><ymax>28</ymax>
                  </bndbox></object>
                </annotation>
                """
            )

            coco, summary = convert(source, "example")

        self.assertEqual(len(coco["images"]), 1)
        self.assertEqual(coco["annotations"][0]["bbox"], [0.0, 10.0, 20.0, 20.0])
        self.assertTrue(coco["annotations"][0]["edge_clipped"])
        self.assertEqual(summary["annotations"], 2)
        self.assertEqual(summary["annotations_by_class"], {
            "pore": 1,
            "stomata": 1,
        })
        self.assertEqual(summary["clipped_boxes"], 1)
        self.assertEqual(summary["rejected_degenerate_boxes"], 1)
        self.assertEqual(summary["filename_metadata_mismatches"], 0)
        self.assertEqual(summary["dimension_metadata_mismatches"], 0)

    def test_archive_renaming_is_recorded_not_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            Image.new("RGB", (100, 80)).save(source / "renamed.jpg")
            (source / "renamed.xml").write_text(
                """
                <annotation>
                  <filename>original-name.jpg</filename>
                  <size><width>100</width><height>80</height></size>
                  <object><class>stomata</class><bndbox>
                    <xmin>10</xmin><ymin>10</ymin>
                    <xmax>20</xmax><ymax>30</ymax>
                  </bndbox></object>
                </annotation>
                """
            )
            coco, summary = convert(source, "example")

        self.assertEqual(coco["images"][0]["file_name"], "renamed.jpg")
        self.assertEqual(
            coco["images"][0]["source_original_filename"],
            "original-name.jpg",
        )
        self.assertEqual(summary["filename_metadata_mismatches"], 1)

    def test_decoded_dimensions_override_incorrect_xml_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            Image.new("RGB", (120, 90)).save(source / "sample.jpg")
            (source / "sample.xml").write_text(
                """
                <annotation>
                  <filename>sample.jpg</filename>
                  <size><width>100</width><height>80</height></size>
                  <object><class>stomata</class><bndbox>
                    <xmin>90</xmin><ymin>70</ymin>
                    <xmax>115</xmax><ymax>85</ymax>
                  </bndbox></object>
                </annotation>
                """
            )
            coco, summary = convert(source, "example")

        image = coco["images"][0]
        self.assertEqual((image["width"], image["height"]), (120, 90))
        self.assertEqual(
            (image["annotation_width"], image["annotation_height"]),
            (100, 80),
        )
        self.assertEqual(coco["annotations"][0]["bbox"], [90.0, 70.0, 25.0, 15.0])
        self.assertEqual(summary["dimension_metadata_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
