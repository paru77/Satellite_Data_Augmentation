import os
import logging
from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, bbox_to_dimensions, MimeType
from datetime import date, timedelta
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# Set up your Sentinel Hub config
config = SHConfig()
config.sh_client_id = '551feea3-d4ab-4127-8812-f67ccfb57ee1'
config.sh_client_secret = 'yIppXilINo7cyFx7Z5uMB1Xexv0kMzXi'


  from sentinelhub import SentinelHubRequest, DataCollection, MimeType, BBox, CRS, bbox_to_dimensions
from PIL import Image
import numpy as np
import pandas as pd
import os
import logging

def get_satellite_data(coords, start_date, end_date, resolution, image_dir, data_type):
    bbox = BBox(bbox=coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    print(f"Image shape at {resolution} m resolution: {size} pixels")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    os.makedirs(image_dir, exist_ok=True)
    if data_type=="natural":
        data_collection = DataCollection.SENTINEL2_L1C
        evalscript = """
        //VERSION=3
        function setup() {

            return {
                input: ["B02", "B03", "B04"],
                output: { bands: 3 }
            };
        }
        function evaluatePixel(sample) {
            let gain = 2.5;
            return [sample.B04, sample.B03, sample.B02];

        }
        """




    if data_type == "aerosol":
        data_collection = DataCollection.SENTINEL5P
        evalscript = """
        //VERSION=3
        var minVal = -1.0;
        var maxVal = 5.0;
        var diff = maxVal - minVal;
        const map = [
          [minVal, 0x00007f],
          [minVal + 0.125 * diff, 0x0000ff],
          [minVal + 0.375 * diff, 0x00ffff],
          [minVal + 0.625 * diff, 0xffff00],
          [minVal + 0.875 * diff, 0xff0000],
          [maxVal, 0x7f0000]
        ];

        const visualizer = new ColorRampVisualizer(map);

        function setup() {
          return {
            input: ["AER_AI_340_380", "dataMask"],
            output: { bands: 4 }
          };
        }

        function evaluatePixel(sample) {
          return [...visualizer.process(sample.AER_AI_340_380), sample.dataMask];
        }
        """

    if data_type == "thermal":
        data_collection = DataCollection.SENTINEL3_SLSTR
        evalscript = """
        //VERSION=3
        const blue_red = [
          [273, 0xffffff],
          [274, 0xfefce7],
          [283, 0xfde191],
          [293, 0xf69855],
          [303, 0xec6927],
          [323, 0xaa2d1d],
          [363, 0x650401],
          [473, 0x3d0200]
        ];

        const viz = new ColorRampVisualizer(blue_red);

        function setup() {
          return {
            input: [
              {
                bands: ["F1", "dataMask"],
              }
            ],
            output: [
              { id: "default", bands: 4 },
              { id: "eobrowserStats", bands: 1 },
              { id: "dataMask", bands: 1 }
            ]
          };
        }

        function evaluatePixel(samples) {
          let val = samples.F1;
          val = viz.process(val);
          val.push(samples.dataMask);
          const statsVal = isFinite(samples.F1) ? samples.F1 - 273 : NaN;
          return {
            default: val,
            eobrowserStats: [statsVal],
            dataMask: [samples.dataMask]
          };
        }
        """


    if data_type == "optical_thickness":
        data_collection = DataCollection.SENTINEL5P
        evalscript = """
        //VERSION=3
        var minVal = 0.0;
        var maxVal = 250.0;
        var diff = maxVal - minVal;
        const map = [
            [minVal, 0x00007f],
            [minVal + 0.125 * diff, 0x0000ff],
            [minVal + 0.375 * diff, 0x00ffff],
            [minVal + 0.625 * diff, 0xffff00],
            [minVal + 0.875 * diff, 0xff0000],
            [maxVal, 0x7f0000]
        ];

        const visualizer = new ColorRampVisualizer(map);

        function setup() {
            return {
                input: ["CLOUD_OPTICAL_THICKNESS","dataMask"],
                output: [
                    { id: "default", bands: 4 },
                    { id: "eobrowserStats", bands: 1 },
                    { id: "dataMask", bands: 1 },
                ],
            };
        }

        function evaluatePixel(samples) {
            const [r, g, b] = visualizer.process(samples.CLOUD_OPTICAL_THICKNESS);
            const statsVal = isFinite(samples.CLOUD_OPTICAL_THICKNESS) ? samples.CLOUD_OPTICAL_THICKNESS : NaN;
            return {
                default: [r, g, b, samples.dataMask],
                eobrowserStats: [statsVal],
                dataMask: [samples.dataMask],
            };
        }
        """

    if data_type == "moisture":
        data_collection = DataCollection.SENTINEL2_L2A
        evalscript = """
        //VERSION=3
        const moistureRamps = [
            [-0.8, 0x800000],
            [-0.24, 0xff0000],
            [-0.032, 0xffff00],
            [0.032, 0x00ffff],
            [0.24, 0x0000ff],
            [0.8, 0x000080]
        ];

        const viz = new ColorRampVisualizer(moistureRamps);

        function setup() {
          return {
            input: ["B8A", "B11", "SCL", "dataMask"],
            output: [
              { id: "default", bands: 4 },
              { id: "index", bands: 1, sampleType: "FLOAT32" },
              { id: "eobrowserStats", bands: 2, sampleType: "FLOAT32" },
              { id: "dataMask", bands: 1 },
            ],
          };
        }

        function evaluatePixel(samples) {
          let val = index(samples.B8A, samples.B11);
          const indexVal = samples.dataMask === 1 ? val : NaN;
          return {
            default: [...viz.process(val), samples.dataMask],
            index: [indexVal],
            eobrowserStats: [val, isCloud(samples.SCL) ? 1 : 0],
            dataMask: [samples.dataMask],
          };
        }

        function isCloud(scl) {
          if (scl == 3) return false;
          else if (scl == 9) return true;
          else if (scl == 8) return true;
          else if (scl == 7) return false;
          else if (scl == 10) return true;
          else if (scl == 11) return false;
          else if (scl == 1) return false;
          else if (scl == 2) return false;
          return false;
        }
        """


    if data_type == "chlorophyll":
        data_collection = DataCollection.SENTINEL3_OLCI
        evalscript = """
        //VERSION=3
        const map = [
          [0.0, 0x00007d],
          [1.0, 0x004ccc],
          [1.8, 0xff3333],
          [2.5, 0xffe500],
          [4.0, 0x00cc19],
          [4.5, 0x00cc19],
          [5.0, 0xffffff],
        ];

        const visualizer = new ColorRampVisualizer(map);
        function setup() {
          return {
            input: ["B10", "B11", "B12", "dataMask"],
            output: [
              { id: "default", bands: 4 },
              { id: "index", bands: 1, sampleType: "FLOAT32" },
              { id: "eobrowserStats", bands: 1 },
              { id: "dataMask", bands: 1 },
            ],
          };
        }

        function evaluatePixel(samples) {
          let OTCI = (samples.B12 - samples.B11) / (samples.B11 - samples.B10);
          let imgVals = null;
          const indexVal =
            samples.dataMask === 1 && OTCI >= -10 && OTCI <= 10 ? OTCI : NaN;
          imgVals = [...visualizer.process(OTCI), samples.dataMask];
          return {
            default: imgVals,
            index: [indexVal],
            eobrowserStats: [indexVal],
            dataMask: [samples.dataMask],
          };
        }
        """


    for date in dates:
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=(date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=bbox,
            size=size,
            config=config
        )

        try:
            data = request.get_data()[0]
            # print(f"Data shape for {date}: {data}")
            image_filename = os.path.join(image_dir, f'{date.strftime("%Y-%m-%d")}_{data_type}.PNG')
            # print(f"Image filename: {data}")
            Image.fromarray((data).astype(np.uint8)).save(image_filename)   #there has to be only data here for natural
            print(f"Saved image: {image_filename}")

        except Exception as e:
            logging.warning(f"Failed to retrieve {data_type} image for {date}: {e}")


# Define coordinates and date range
latitude =47.600968892668575
longitude = 8.183936440913271
buffer = 0.02 # Small buffer for bounding box
coords = (longitude - buffer, latitude - buffer, longitude + buffer, latitude + buffer)

image_dir="/DS/dsg-ml/work/pnegi/dataset/leibstadt/thermal"
get_satellite_data(coords, "2023-01-01", "2023-12-31", resolution=10, image_dir=image_dir, data_type="thermal")
image_dir= '/DS/dsg-ml/work/pnegi/dataset/leibstadt/natural'
get_satellite_data(coords, "2023-01-01", "2023-01-02", resolution=10, image_dir=image_dir, data_type="natural")
image_dir= '/DS/dsg-ml/work/pnegi/dataset/leibstadt/optical_thickness'
get_satellite_data(coords, "2023-01-01", "2023-12-31", resolution=10, image_dir=image_dir, data_type="optical_thickness")
image_dir='/DS/dsg-ml/work/pnegi/dataset/leibstadt/moisture'
get_satellite_data(coords, "2023-01-01", "2023-01-02", resolution=10, image_dir=image_dir, data_type="moisture")
image_dir="/DS/dsg-ml/work/pnegi/dataset/leibstadt/chlorophyll"
get_satellite_data(coords, "2023-01-01", "2023-12-31", resolution=10, image_dir=image_dir, data_type="chlorophyll")