
## AI Cybersecurity Platform

This project is a comprehensive web-based survey platform designed to employ LLM to assess the cybersecurity readiness of organizations. It also includes functionalities for phishing detection, suspicious website detection, and cybersecurity trends analysis. 

### Environment
- Python 3.10.14

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/periscopegithub/Capstone-AI-Cybersecurity-Platform.git
    cd Capstone-AI-Cybersecurity-Platform
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Requirements

#### 1. API Keys
This project utilizes Azure's OpenAI API. You need to create a `.env` file in the root directory and add your API keys:

```
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_KEY=your_key_here
AZURE_OPENAI_VERSION=your_version_here
AZURE_OPENAI_DEPLOYMENT_GPT35TURBO=your_gpt35turbo_deployment_here
AZURE_OPENAI_DEPLOYMENT_GPT4=your_gpt4_deployment_here
AZURE_OPENAI_DEPLOYMENT_GPT4O=your_gpt4o_deployment_here
AZURE_OPENAI_EMBEDDING_MODEL=your_embedding_model_here
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_here

```

#### 2. Ollama
The scripts utilize Ollama for running local LLM. You need to install Ollama. Refer to the [Ollama website](https://ollama.com/) for installation instructions.

#### 3. Llama3 Model
Download the Llama3 model in Ollama for local LLM operations.

#### 4. CUDA and cuDNN
For running the Suspicious Website Detector, ensure you have an Nvidia GPU with CUDA toolkit and cuDNN installed. Refer to the official [Nvidia CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) websites for installation instructions.

### Steps to Start

1. **Start Redis**:
   - Execute `start.bat` in the Redis folder, or download and start Redis from the [official website](https://redis.io/).

2. **Start Celery**:
   - In a terminal, navigate to the project directory and run:
     ```sh
     celery -A celery_app.celery worker --pool=solo --loglevel=info
     ```

3. **Run the Survey Application**:
   - Execute the following command to start the Flask application:
     ```sh
     python survey.py
     ```

4. **Access the Web UI**:
   - Open a web browser and navigate to [http://127.0.0.1:5000/welcome](http://127.0.0.1:5000/welcome) to start using the platform.

### Usage

- **Welcome Page**: Provides an overview of the platform and the Hong Kong Cybersecurity Index.
- **Survey Page**: Users can enter their invitation code to start the survey.
- **Phishing Prevention**: Includes tools for detecting suspicious emails and websites.
- **Cyber Threat Trends**: Displays recent cyber threat data and trends.

### Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

### License

This project is licensed under the MIT License. See the LICENSE file for more details.
