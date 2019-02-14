This is my first SLAM system which is actually based on ORB_SLAM2 but without loop-closing algorithm.  (There is some explaination of the whole algorithm in the code.)
I tried my project on TUM dataset.

## TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py rgbd_datatest/rgb.txt rgbd_datatest/depth.txt > associations.txt

  python associate.py fr1_desk/rgb.txt fr1_desk/depth.txt > associations.txt

  python associate.py rgbd_desk/rgb.txt rgbd_desk/depth.txt > associations.txt

  python associate.py rgbd_dataset_slam/rgb.txt rgbd_dataset_slam/depth.txt > associations.txt

  python associate.py fr1_xyz/rgb.txt fr1_xyz/depth.txt > associations.txt

  ```

3. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the path to the coresponding associations file.

  ```
  ./main_tum setting/TUM1.yaml Vocabulary/ORBvoc.txt rgbd_datatest associations.txt

  ./main_tum setting/TUM1.yaml Vocabulary/ORBvoc.txt fr1_desk associations.txt

  ./main_tum setting/TUM1.yaml Vocabulary/ORBvoc.txt rgbd_desk associations.txt

  ./main_tum setting/TUM1.yaml Vocabulary/ORBvoc.txt rgbd_dataset_slam associations.txt

  ./main_tum setting/TUM1.yaml Vocabulary/ORBvoc.txt fr1_xyz associations.txt

#######
If u want to run the code in qtcreator， just use the cmd starting from setting/... without main

fr1/desk2 for desk_test2
fr2/desk

Drawing trajactory
python draw_groundtruth.py

