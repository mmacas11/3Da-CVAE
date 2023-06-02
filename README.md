# An Attention-Based Deep Generative Model for Anomaly Detection in Industrial Control Systems
Anomaly detection is critical for the secure and reliable operation of industrial control systems. As we rely more on such complex cyber-physical systems, it becomes paramount to have automated methods for detecting anomalies, preventing attacks, and responding intelligently. This paper presents a novel deep generative model called 3Da-CVAE to address this need. The proposed model uses a variational autoencoder with a 3D convolutional encoder and decoder to extract features from both spatial and temporal dimensions. We also incorporate an attention mechanism that directs focus towards specific regions, enhancing the representation of relevant features and improving anomaly detection accuracy. We use a dynamic threshold approach with the reconstruction probability and make our source code publicly available to promote reproducibility and facilitate further research. Comprehensive experimental analysis is conducted on data from all six stages of the Secure Water Treatment (SWaT) testbed, and the experimental results demonstrate the superior performance of our approach compared to several state-of-the-art baseline techniques.

## Table Of Contents
-  [In Details](#in-details)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)
- [License](#license)

## In Details
- DoVaeSupplementary: this file contains the class Sampling, class VAE and class EarlyStoppingAtMinLoss for implementing a VAE model.
- DoSupplementaryTrain.py: diferentes métodos para guardar información cuando se ejecuta el entrenamiento del modelo.
- DoSplitData.py: data prepatation and split data.
- CBAM_attention3D.py: this file contains the class channel attention, class spatial attention for implementation of 3D-CBAM.
- 3Da-CVAE.ipynb: model 3-dimensional Attention-Based Convolutional Variational Autoencoder.

## Contributing
Any kind of enhancement or contribution is welcomed.

## Acknowledgments
The authors would like to thank iTrust, Center for Research in Cyber Security, Singapore University of Technology and Design for providing the SWaT dataset.


## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](https://github.com/mmacas11/3Da-CVAE/blob/main/LICENSE.md)**



