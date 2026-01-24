#!/bin/bash

echo "🌱 Seeding Sample Data"
echo "====================="

# Sample documents about machine learning
curl -X POST http://localhost:8000/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {
        "content": "Machine learning (ML) is a branch of artificial intelligence (AI) and computer science that focuses on using data and algorithms to imitate the way humans learn, gradually improving accuracy. Machine learning is an important component of the growing field of data science.",
        "metadata": {"source": "intro", "topic": "ML Basics"}
      },
      {
        "content": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The model makes predictions based on input data and is corrected when those predictions are wrong. Common algorithms include linear regression, logistic regression, decision trees, and neural networks.",
        "metadata": {"source": "supervised", "topic": "ML Types"}
      },
      {
        "content": "Unsupervised learning is used when we have unlabeled data. The algorithm tries to find patterns and relationships in the data without any guidance. Common techniques include clustering (like K-means), dimensionality reduction (like PCA), and association rules.",
        "metadata": {"source": "unsupervised", "topic": "ML Types"}
      },
      {
        "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks). It has revolutionized fields like computer vision, natural language processing, and speech recognition. Popular frameworks include TensorFlow, PyTorch, and Keras.",
        "metadata": {"source": "deep-learning", "topic": "Advanced ML"}
      },
      {
        "content": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It includes tasks like text classification, sentiment analysis, machine translation, and question answering. Modern NLP relies heavily on transformer models like BERT and GPT.",
        "metadata": {"source": "nlp", "topic": "NLP"}
      },
      {
        "content": "Model evaluation is crucial in machine learning. Common metrics include accuracy, precision, recall, F1-score for classification, and MSE, RMSE, MAE for regression. Cross-validation techniques like k-fold are used to assess how well a model generalizes to unseen data.",
        "metadata": {"source": "evaluation", "topic": "ML Best Practices"}
      },
      {
        "content": "Feature engineering is the process of selecting, manipulating, and transforming raw data into features that can be used in supervised learning. Good features can significantly improve model performance. Techniques include scaling, encoding categorical variables, and creating interaction terms.",
        "metadata": {"source": "features", "topic": "ML Best Practices"}
      },
      {
        "content": "Overfitting occurs when a model learns the training data too well, including its noise and outliers, leading to poor performance on new data. Regularization techniques like L1 (Lasso) and L2 (Ridge) help prevent overfitting by adding penalty terms to the loss function.",
        "metadata": {"source": "overfitting", "topic": "ML Challenges"}
      },
      {
        "content": "Transfer learning is a technique where a model developed for one task is reused as the starting point for a model on a second task. This is especially useful in deep learning where training from scratch requires large datasets and computational resources.",
        "metadata": {"source": "transfer", "topic": "Advanced ML"}
      },
      {
        "content": "MLOps (Machine Learning Operations) is a set of practices that combines ML, DevOps, and data engineering to deploy and maintain ML systems in production reliably and efficiently. It includes model versioning, monitoring, and automated retraining pipelines.",
        "metadata": {"source": "mlops", "topic": "ML Production"}
      }
    ]
  }'

echo ""
echo "✅ Sample data seeded successfully!"
echo ""
echo "Try asking questions like:"
echo "  - What is machine learning?"
echo "  - Explain supervised learning"
echo "  - What is overfitting?"
echo "  - Tell me about deep learning"
