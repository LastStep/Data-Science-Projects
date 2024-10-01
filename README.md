# Projects Developed During Internship at SocialPrachar (Hyderabad)

A collection of data science and machine learning projects I developed during my internship at **SocialPrachar**, Hyderabad. These projects focus on diverse domains, including employee attrition analysis, face detection, and product recommendation, with a goal to showcase various techniques in machine learning and data processing.

## 🔍 Overview of Projects

### 1. **Employee Attrition Analysis**
- **Description**: This project aims to predict **employee attrition** using machine learning techniques. It starts with an in-depth **Exploratory Data Analysis (EDA)** to understand the data and extract insights about the factors affecting employee turnover.
- **Technologies Used**: The project employs a **Random Forest Classifier** to predict whether an employee will leave or stay, based on factors like age, job role, satisfaction level, etc.
- **Skills Gained**: Hands-on experience with data preprocessing, feature selection, and applying classification algorithms to solve real-world HR challenges.

### 2. **FaceNet Face Detection**
- **Description**: This project demonstrates **face detection and recognition** using the **FaceNet model**, a state-of-the-art deep learning architecture for facial recognition and feature extraction. The primary goal is to detect faces in images and extract embeddings for further identification or verification purposes.
- **Technologies Used**:
  - **FaceNet Model**: Utilized for high-accuracy face detection, feature extraction, and facial recognition tasks.
  - **Google Sheets API**: Implemented to store security logs, which can be used to maintain records of detected individuals for security purposes.
- **Skills Gained**: Working with neural networks for computer vision tasks, using APIs for data logging, and understanding face embeddings for recognition.

### 3. **Product Recommendation System**
- **Description**: This project focuses on building a **Product Recommendation System** that suggests products to users based on their previous purchases and interests. It uses natural language processing techniques to generate product embeddings, helping recommend similar items.
- **Technologies Used**:
  - **Doc2Vec**: A variant of Word2Vec used for creating vector representations of product descriptions. This helps the model understand the relationships between different products and make recommendations.
- **Skills Gained**: Experience with recommendation systems, applying **Doc2Vec** to create vector representations, and building a content-based recommendation system for e-commerce.

### 4. **User Product Recommendation System**
- **Description**: This project is an integration of the previous projects into a comprehensive system that provides personalized product recommendations to customers based on face detection and previous purchase history. 
  - It begins with using the **Face Detection Model** (FaceNet) to identify customers in real-time.
  - Once a customer is identified, the **Product Recommendation System** is used to recommend products tailored to their preferences and purchase history.
  - **Google Firebase** is used to securely store customer data, including personal details and purchase logs.
- **Technologies Used**:
  - **FaceNet** for real-time face recognition of customers.
  - **Google Firebase** for storing customer data and managing secure access.
  - **Product Recommendation System** from Project 3 for suggesting products to the detected customers.
- **Skills Gained**: Combining multiple machine learning models, using cloud-based databases for data storage, and delivering an end-to-end product recommendation solution.

## 🚀 Key Takeaways
- Developed a **full-stack machine learning solution** that integrates multiple models and techniques for real-world applications.
- Enhanced understanding of data preprocessing, **EDA**, and model selection.
- Gained expertise in deploying cloud-based services like **Google Sheets API** and **Google Firebase** for data management.
- Applied **Doc2Vec** for product recommendation, and successfully leveraged **deep learning** models like **FaceNet** for face detection and recognition.
