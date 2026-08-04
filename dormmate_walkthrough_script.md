# Video Walkthrough Script: DormMate Backend & Algorithmic Core
**Target Audience**: Mayank Mundhra & The Agentropic Team  
**Estimated Video Duration**: 4 - 5 minutes  
**Goal**: Demonstrate backend design, data processing pipelines, ML/DL matching mechanics, and showcase senior-level engineering foresight with advanced modern improvements (LLMs, RL, GNNs, and Vector DBs).

---

## 🎬 Pre-Recording Setup & Tips
* **IDE Preparation**: 
  * Open two tabs in VS Code or your preferred editor: [main_script.py](file:///c:/Users/mohak/OneDrive/Documents/Projects/Roommate-Compatibility-Algorithm/app/core/main_script.py) and [endpoints.py](file:///c:/Users/mohak/OneDrive/Documents/Projects/Roommate-Compatibility-Algorithm/app/endpoints/endpoints.py).
  * Zoom in slightly so code text is crisp and readable on a 1080p/4K screen.
* **Database & Server**:
  * Have your FastAPI local dev server running in a terminal.
  * Optionally, have MongoDB Compass or Atlas open in the background to show the database collection structure.
* **Recording Flow**:
  * Keep your mouse cursor steady. When explaining code, highlight the specific lines you are talking about.
  * Speak at a steady, conversational pace. Don't rush; let your confidence show.

---

## 🎙️ Script Walkthrough

### Part 1: The Hook & Introduction (~45 seconds)
**Visual**: Show your IDE with [app/app.py](file:///c:/Users/mohak/OneDrive/Documents/Projects/Roommate-Compatibility-Algorithm/app/app.py) or the main README open. Highlight the file structure.

* **[What to Say]**:
  > "Hi Mayank and the Agentropic Team, Mohak here. Thanks for giving me the opportunity to walk you through **DormMate**, a roommate matching and compatibility engine I built.
  > 
  > DormMate was designed to move beyond arbitrary manual roommate matching by using data-driven algorithms to pair students. 
  > Today, I'll walk you through the backend architecture, our dual-track matching pipeline, and then wrap up with how I would re-architect and level up this project today using LLMs, Reinforcement Learning, and modern vector databases."

---

### Part 2: Data Ingestion & preprocessing Pipeline (~60 seconds)
**Visual**: Switch to [app/core/main_script.py](file:///c:/Users/mohak/OneDrive/Documents/Projects/Roommate-Compatibility-Algorithm/app/core/main_script.py). Scroll to lines 283–384 (`fetch_google_sheet`, `clean_data`, and `encode_data`). Highlight these functions.

* **[What to Say]**:
  > "Let's start with how the data enters the system. 
  > We ingest questionnaire data from Google Sheets using `gspread`. Because raw survey data is notoriously messy, the pipeline cleans and standardizes it.
  > 
  > Under `clean_data`, we map conversational column headers to structured fields, impute missing values, and parse open text fields like hobbies.
  > 
  > Under `encode_data`, we convert categorical and ordinal data into machine-readable arrays. We use manual mapping for ordinal features like bedtime schedules or cleaning frequency to preserve their natural order. For categorical values like study location and hobbies, we apply One-Hot Encoding via scikit-learn.
  > 
  > One critical detail here is how we calculate feature importance. We load a `question_mapping.csv` containing normalized weights for each question. We multiply our raw feature matrix `X` by these weights before standardizing with a `StandardScaler`. This ensures that high-impact compatibility questions, like sleep schedule or noise tolerance, carry more weight in the similarity calculations than low-impact ones."

---

### Part 3: The Dual Matching Engine (~90 seconds)
**Visual**: Scroll to the training and testing functions in [app/core/main_script.py](file:///c:/Users/mohak/OneDrive/Documents/Projects/Roommate-Compatibility-Algorithm/app/core/main_script.py). Highlight `train_model` (lines 70–92), `test_with_model` (lines 170–191), and `test_with_clusters` (lines 193–224).

* **[What to Say]**:
  > "Depending on the dataset size and label availability, the backend employs a dual-track architecture:
  > 
  > **First, the Deep Learning Track**: When we have historical compatibility ratings to train on, we train a Multi-Layer Perceptron using TensorFlow and Keras. 
  > The network takes the scaled, weighted roommate features and learns to predict a compatibility score. 
  > Once trained, we use the neural network as a feature representation generator. To find matches, we pass user feature vectors through the model to obtain embeddings, and compute roommate compatibility as the inverse Euclidean distance between their embedding representations.
  > 
  > **Second, the Clustering Fallback Track**: If the dataset is small—under 500 records—training a deep neural network would overfit. In this case, we fall back to a K-Means clustering pipeline. 
  > We group students into cohorts using K-Means, and then within each cohort, compute matches using a custom weighted Manhattan similarity:
  > `weighted_similarity = 1 - (sum(weight * |user - others|) / sum(weights))`.
  > 
  > This hybrid approach guarantees that we get high-quality matches even when bootstrap data is limited."

---

### Part 4: API Layers & Storage (~30 seconds)
**Visual**: Switch to [app/endpoints/endpoints.py](file:///c:/Users/mohak/OneDrive/Documents/Projects/Roommate-Compatibility-Algorithm/app/endpoints/endpoints.py). Scroll through the API router definitions and show `api_signup` / `api_login` calling the MongoDB service.

* **[What to Say]**:
  > "All of this logic is exposed through a REST API built with FastAPI. We choose FastAPI for its asynchronous performance, automatic Swagger docs generation, and Pydantic validation.
  > 
  > For data storage and authentication, we interface with MongoDB. The service layer handles user registration and logins, using `bcrypt` to hash and verify passwords securely. This keeps user credentials isolated and secure from the raw roommate dataset."

---

### Part 5: Retrospective & Future Improvements (~90 seconds)
**Visual**: Open a new text file or draw a simple diagram, or just highlight your face if you are on camera. Alternatively, highlight the future improvement section of this script. 

* **[What to Say]**:
  > "Looking back at this project, it was a fantastic exercise in structuring a data-to-model pipeline. But knowing what I know now—and given the advancements in AI—here is how I would re-architect the backend for a production-grade system today:
  > 
  > 1. **LLM-based Semantic Tagging & Embeddings**:
  >    Instead of restricting users to static category lists or doing primitive comma-splitting for hobbies, I would use an LLM like Gemini or Claude to parse open-ended user bios. The LLM could extract fine-grained interest tags and generate rich semantic profile embeddings using models like OpenAI's `text-embedding-3`. This captures nuanced compatibility that one-hot encoding misses.
  > 
  > 2. **Reinforcement Learning from Feedback (RL)**:
  >    Compatibility isn't static. In a live app, I'd model roommate matching as a recommendation system. Using implicit feedback—like profile clicks, chat initiation, or swipe-left/swipe-right decisions—we could train a Contextual Bandit or Reinforcement Learning agent to continuously refine and adapt to individual preferences and subjective 'taste' over time.
  > 
  > 3. **Graph Neural Networks (GNNs)**:
  >    Roommates don't live in a vacuum; they live in a social graph. I'd represent the student base as a heterogeneous graph where nodes are users and edges represent mutual friends, classes, majors, or dorm floors. By training a GNN like GraphSAGE, we could propagate neighbor information to generate embeddings that capture both individual preferences and social network context.
  > 
  > 4. **Production Vector Database (ANN Search)**:
  >    Currently, similarity is computed in memory with numpy, which is an $O(N)$ operation. In production, I would load the user embeddings into a vector database like Qdrant or Pinecone. We could perform Approximate Nearest Neighbor searches using HNSW indexes, achieving sub-millisecond retrieval times scaling to millions of users."

---

### Part 6: Outro (~20 seconds)
**Visual**: Switch back to the main file or your contact info.

* **[What to Say]**:
  > "That is a quick high-level overview of the DormMate backend. I'm excited about building intelligent, agentic systems, and I'd love to bring this combination of machine learning theory and practical engineering to the Agentropic Team. 
  > Thanks for watching, and I look forward to our call!"

---

## 💡 Key Terms Cheat-Sheet (For Q&A preparation)
* **StandardScaler**: Centers data to 0 mean and 1 variance, ensuring that different question ranges (e.g., 1–10 vs. Yes/No binary) don't skew distance calculations.
* **K-Means Clustering**: Unsupervised grouping algorithm. Used here to segment the user base into smaller, highly compatible groups before ranking them individually.
* **Multi-Layer Perceptron (MLP)**: Feedforward neural network used to learn non-linear patterns of compatibility based on historical scoring.
* **Euclidean vs. Manhattan Distance**: Euclidean (L2 norm) represents straight-line distance in vector space (ideal for continuous embeddings). Manhattan (L1 norm) summates absolute difference across categories, which is highly interpretable when calculating attribute-by-attribute offsets.
