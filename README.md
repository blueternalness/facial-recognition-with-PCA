# [Facial Recognition with PCA](PDF/Facial_recognition_with_PCA.pdf)

We initiated this project from the challenge: “the curse of Dimensionality”.
Even a small face image have a massive vector with 3,000 dimensions and processing this high-dimensional pixel space is computationally inefficient.

Based on this Challenge, we came to the insight. Faces are not random noise. They are structured objects. For example, all faces have two eyes and ears. So, we assumed the set of valid human faces resides on a much lower-dimensional subspace.

Also, we found the limitation of the existing limitation of Deep Learning for facial recognition. Neural Networks are computationally heavy and they require  massive datasets and GPUs for training and inference. So, they are often unsuitable for low-power, real-time edge devices or embedded systems.

Finally we came to the solution: Lightweight PCA & Eingefaces. We used Eigendecomposition as a feature extraction engine. Our goal was to develop a CPU-friendly system capable of facial recognition.


## Link to the project report
[A Detailed Report on Our Work](PDF/Facial_recognition_with_PCA.pdf)
