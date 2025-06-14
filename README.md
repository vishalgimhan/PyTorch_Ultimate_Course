# PyTorch_Ultimate_Course

**Neural Network Fundamentals**

- Core Concepts: Perceptrons, simple and deep neural networks.
- Layers: Input, Dense, 1D Convolutional, RNN, LSTM, Output (Regression, Multi-Target, Binary/Multi-Label Classification).
- Activation Functions: ReLU, Leaky ReLU, tanh, Sigmoid, Softmax.
- Loss Functions: MSE, MAE, MBE (Regression); Binary Cross Entropy, Hinge Loss, Multi-Label Cross Entropy (Classification).
- Optimization: Gradient Descent, Learning Rate, Optimizers (Adagrad, Adam, SGD, BGD).
- Challenges: Address underfitting (high bias) and overfitting (high variance).

**Building Neural Networks from Scratch**

- Key Processes: Forward pass, backpropagation, chain rule, dot product.
- Tensors: Core data structures for neural networks.
- PyTorch Training: Clear gradients, forward pass, loss calculation, gradient computation, weight updates.
- Implementation: Linear regression and custom model classes.

**Data Handling & Training**

- Data Management: Batches, Datasets, Dataloaders, Model Saving/Loading.
- Hyperparameter Tuning: Network topology (nodes, layers, activation), loss, optimizer, learning rate, batch size, epochs using Skorch.

**Classification Models**

- Types: Binary, Multi-Class (mutually exclusive), Multi-Label (not mutually exclusive).
- Evaluation: Confusion Matrix, ROC, Threshold, TPR, FPR, AUC.

**Image Classification with CNNs**

- Components: Convolutional filters (edge detection, blur), feature maps, max pooling.
- Preprocessing: Resize, CenterCrop, Grayscale, RandomRotation, RandomVerticalFlip, ToTensor, Normalize.
- Calculations: Tensor dimension management.

**Audio Classification**

- Approaches: Treat audio as images or time series for classification.

**Object Detection**

- Metrics: IoU, Precision-Recall Curve, Mean Average Precision (mAP).
- Algorithms: Two-Stage (Fast R-CNN, Faster R-CNN), One-Stage (YOLO, SSD).
- Tools: Detecto, YOLOv7/YOLOv8 on Colab with GPU.
- Labeling Formats: PASCAL VOC, COCO, YOLO.

**Style Transfer**

- Techniques: Combine content and style images using pretrained VGG16/VGG19.
- Losses: Content Loss, Style Loss, Feature Map Correlation.

**Pretrained Models & Transfer Learning**

- Models: DenseNet and other pretrained networks for efficient training.

**RNNs & Time Series**

- Applications: Time Series, NLP, Speech Recognition.
- Architectures: Rolled/Unrolled RNNs, LSTMs.

**Recommender Systems**

- Types: Content-Based, Collaborative Filtering (Item/User CF, Matrix Factorization).
- Evaluation: Precision@k, Recall@k.

**Autoencoders & GANs**

- Autoencoders: Shallow and Deep architectures with Encoder/Decoder.
- GANs: Generator, Discriminator, and associated losses.

**Graph Neural Networks (GNNs)**

- Concepts: Graphs, Dense/Sparse Matrices, Non-Euclidean Spaces.
- Tasks: Link Prediction, Graph/Node Classification.

**Transformers**

- Components: Positional encoding, attention, self-attention.
- Models: BERT, GPT-3, LaMDA, Vision Transformers (ViT).

**PyTorch Lightning**

- Workflow: __init__, forward, configure_optimizers, training_step, validation_step, test_step, predict_step.
Features: Early Stopping.

**Semi-Supervised Learning**

- Combine labeled and unlabeled data for robust model training.

**Natural Language Processing (NLP)**

- Tasks: Next word prediction, sentiment analysis, classification, translation.
- Vocabulary: Tokenization, Tokens, Documents, Corpus.
- Word Embeddings: One-Hot Encoding, Frequency-Based (Count, TF-IDF, Co-Occurrence), Neural Network-Based (Word2Vec, GloVe, BERT, GPT).
- Applications: Sentiment models, pretrained Hugging Face models, zero-shot classification, natural language inference.

**Advanced Topics**

- Architectures: ResNet (skip connections), Inception, Extreme Learning Machines (ELM).
- Applications: Image similarity, vector databases (Pinecone, Chroma, Redis), Retrieval Augmented Generation (RAG).
- LLMs: Claude by Anthropic (Haiku, Sonnet, Opus), agents (memory, tools, planning, CrewAI, LangGraph).
- LLM Optimization: Prompt engineering, RAG, finetuning.

**Model Debugging**

- Techniques: Debug CNNs using hooks for better insights.

**Deployment**

- Platforms: On-premise, Cloud (GCP, AWS, Azure).
- APIs: REST, GraphQL, gRPC; URL anatomy, HTTP methods (GET, POST, DELETE, PUT).
- Tools: Flask, Postman, Google Cloud for deploying weights and functions.
