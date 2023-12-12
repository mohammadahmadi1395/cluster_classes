
  # MixMaxSim: Mixture of MaxMax and Similarity 📝  
  MixMaxSim project is mainly maintained By [Sayed Mohammad Ahmadi](mailto:ahmadi.mohammad2008@gmail.com) and [Ruhollah Dianat](mailto:ruhollah.dianat@gamil.com). </br></br>
  This project implements a framework in face identification which uses divide and conqure idea and add an intelligent post-processing to predict better.</br></br>
  This implementation is related to a manuscript which is under review in [IET Computer Vision Journal](https://ietresearch.onlinelibrary.wiley.com/journal/17519640) and is under the divide and conqure approach, such as [Independent Softmax Model](https://dl.acm.org/doi/abs/10.1145/2964284.2984060) paper and improves it by two modifications: using clustering instead of random distribution and using intelligent post-processing for combination of submodels. </br></br>
  This framework improves speed and memory and accuracy in comparison to single models.</br></br>
  This project is independent of base models and dataset. </br></br>
  Every one can use this framework for his own dataset or backbone.</br>

  ## Get Started 🚀  
  If you want implement the project in your machine, you need to have dataset, prepare the data and then implement the project.
  
  ## Requiremnts
  python version: 3.7.12
  ``` pip install -r requiremnts.txt```

  ### Download datasets
  We used three datasets. You can download them or use your own dataset:
  1. Glint360k: magnet:?xt=urn:btih:E5F46EE502B9E76DA8CC3A0E4F7C17E4000C7B1E&dn=glint360k
  2. MS-Celeb-1M: Torrent file is located in: ./data/MS-Celeb-1M/MS-Celeb-1M-9e67eb7cc23c9417f39778a8e06cca5e26196a97.torrent
  3. VGGFace2: magnet:?xt=urn:btih:535113b8395832f09121bc53ac85d7bc8ef6fa5b&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce

  ### Prepare data
  <b> IMPORTANT: The following steps are done for all datasets and for instance you can see data preparation for ms-celeb-1m dataset [here](./notebooks/ms1m_data_preparation.ipynb).</b> </br>
  Each dataset will be located in "./data/dataset_name/" </br>
  And includes three directories: train, test, val </br>
  Each directory includes directories with identity names. </br>
  For fair evaluation, for each identity, choose randomly 20 images for train, 5 for test, and 5 for validation. </br>
  For example:</br>
  ```
  ├── glint360k/
  │   ├── train/
  │   │   ├── id_1/
  │   │   │    ├── custom_name_1_1.jpg
  │   │   │    ├── custom_name_1_2.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_1_20.jpg
  │   │   ├── id_2/
  │   │   │    ├── custom_name_2_1.jpg
  │   │   │    ├── custom_name_2_2.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_2_20.jpg
  │   │   ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    └── ...
  │   │   ├── id_360000/
  │   │   │    ├── custom_name_360000_1.jpg
  │   │   │    ├── custom_name_360000_2.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_360000_20.jpg
  │   ├── val/
  │   │   ├── id_1/
  │   │   │    ├── custom_name_1_21.jpg
  │   │   │    ├── custom_name_1_22.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_1_25.jpg
  │   │   ├── id_2/
  │   │   │    ├── custom_name_2_21.jpg
  │   │   │    ├── custom_name_2_22.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_2_25.jpg
  │   │   ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    └── ...
  │   │   ├── id_360000/
  │   │   │    ├── custom_name_360000_21.jpg
  │   │   │    ├── custom_name_360000_22.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_360000_25.jpg
  │   ├── test/
  │   │   ├── id_1/
  │   │   │    ├── custom_name_1_26.jpg
  │   │   │    ├── custom_name_1_27.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_1_30.jpg
  │   │   ├── id_2/
  │   │   │    ├── custom_name_2_26.jpg
  │   │   │    ├── custom_name_2_27.jpg
  │   │   │    ├── ...
  │   │   │    └── custom_name_2_30.jpg
  │   │   ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    ├── ...
  │   │   │    └── ...
  │   │   ├── id_360000/
  │   │   │    ├── custom_name_360000_26.jpg
  │   │   │    ├── custom_name_360000_27.jpg
  │   │   │    ├── ...
  └   └   └    └── custom_name_360000_30.jpg
  ```
  #### Notes: 
  1. If some ids have less than 30 images, you can create new images using augmentation techniques.
  2. Choose the train, test and val images completely random.
  
  ### Feature Extraction
  For speed up, we use features instead of images. So first extract features of all images in dataset using a strong pretrained model. </br>
  Of course it is better that you train a model by scratch.
  For extract features, we used this [onnx pretrained model](https://github.com/onnx/models/blob/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx). ([Reference](https://github.com/onnx/models/tree/main/vision/body_analysis/arcface)) </br>
  By default, we saved extracted features [here](./features/), but you can change the path in [config file](./config/config.json). </br>
  The file structure for features is:
  ```
  ├── glint360k/
  │   ├── train/
  │   │   ├── id_1.npz
  │   │   ├── id_2.npz
  │   │   ├── ...
  │   │   └── id_360000.npz
  │   ├── val/
  │   │   ├── id_1.npz
  │   │   ├── id_2.npz
  │   │   ├── ...
  │   │   └── id_360000.npz
  │   ├── test/
  │   │   ├── id_1.npz
  │   │   ├── id_2.npz
  │   │   ├── ...
  └   └   └── id_360000.npz
  ```
  #### example of feature extraction:
  onnx_model = onnx.load('models/model.onnx')
  tf_rep = prepare(onnx_model) # Import the ONNX model to Tensorflow

  ``` 
  # Crop and save the face
  # Detect faces using RetinaFace
  faces = RetinaFace.extract_faces(img_path=image_path, align=True)
  if len(faces) != 1:
    return
  face_image = faces[0]
  x_train = tf.image.resize(np.array(face_image), (112, 112), method="nearest")
  x_train = (tf.cast(x_train, tf.float32) - 127.5) / 128.
  x_train = tf.transpose(x_train, perm=[2, 0, 1])
  x_train = tf.expand_dims(x_train, 0)
  x_train_emb = tf_rep.run(np.array(x_train))._0
  ```

  [This](notebooks/ms1m_data_preparation.ipynb) notebook file prepares all data and extract features for MSCeleb1M dataset</br>
  And [this](notebooks/vgg_data_preparation.ipynb) one prepares all data and extract features for VGGFace2 dataset </br>

  ### Load the configuration file
  ```
  with open("./config/config.json", "r") as config_file:
    config = json.load(config_file)
  ```

  ### set parameters in config.json file
  ```
  "dataset_name": "vggface2",
  "method": "ISM",  
  "n_classes": 10000, 
  "n_clusters": 5,
  "distance_measure": "cosine", 
  ```
  Notes:
  - dataset_name: "vggface2" or "glint360k" or "ms1m"
  - methods: 
    - MMS (Mixture of Max-max and Similarity) 
    - ISM (Independent Softmax Model)
    - Single (without distribution)
  - n_classes: 
    - vggface2: the max number of classes is 8900
    - glint360k: the max number of classes is 360,000 
    - ms1m: the max number of classes is 100,000
  - distance_measure: "cosine" or "euclidean"
  ### Load the configuration file
  ```
  with open("./config/config.json", "r") as config_file:
    config = json.load(config_file)
  ```
  ### prepare train, test, val datasets based on number of classes
  ```
  from src.data_utility import prepare_data
  trainx, trainy, trainl, \
  traincenterx, traincentery, traincenterl, \
  testx, testy, testl, valx, valy, vall = \ 
                  prepare_data(config)
  ```
  ### Distribute classes among submodules
  ```
  from src.data_utility import cluster_data
  parts = cluster_data(method, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall)
  ```
  ### Train each submodule  
  ```
  from src.data_utility import train_submodels
  train_submodels(method, parts, trainx, trainy, trainl, traincenterx, traincentery, traincenterl, testx, testy, testl, valx, valy, vall)
  ```
  ### result combination
  #### ISM
  ```
  test_softmax_classes = ism_post_process(m, 'test')
  evaluate_ism(m, test_softmax_classes)
  ```
  #### MMS
  ```
  # calculate values based on max-max criterion and distance measure (cosine similarity) criterion on validation dataset
  val_sim_classes, val_sim_values, val_sim_softmax, val_softmax_values, val_softmax_sims, val_softmax_classes = mms_post_process(m, 'val')
  
  # find the point that balances between max-max criterion and cosine similarity criterion on validation dataset
  thr = find_best_thr(val_sim_classes, val_sim_values, val_sim_softmax, val_softmax_values, val_softmax_sims, val_softmax_classes)

  # calculates values based on max_max criterion and distance measure (cosine similarity) criterion on test dataset 
  test_sim_classes, test_sim_values, test_sim_softmax, test_softmax_values, test_softmax_sims, test_softmax_classes = mms_post_process(m, 'test')
  
  # finally evaluates on test dataset
  evaluate_mms(thr, test_sim_classes, test_sim_values, test_sim_softmax, test_softmax_values, test_softmax_sims, test_softmax_classes)
  ```
  #### Single
  for Single method, we can use ISM with one cluster.

  ## All in One
  includes data preparation, distribute classes, train each submodel, combine results and save the evaluation results on file
  ### ISM
  ```
  python3 main_ism.py
  ```

  ### MMS
  ```
  python3 main_mms.py
  ```

  ### Single
  ```
  python3 main_single.py
  ```