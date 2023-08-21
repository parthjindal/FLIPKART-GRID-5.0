# FLIPKART GRID 5.0

PLEASE SET THE OPENAI_API_KEY in the config.py file before running the application.

Requirements:
Install the requirements using the following command:
```
pip install -r requirements.txt
```

Use docker to run the redis stack-server from the folder:
```
docker run --name redis-stack-server -p 6380:6379 -v ./fashion_agent/embeddings:/data redis/redis-stack-server:latest 
```

Run the application using the following command:
```
streamlit run app.py
```


## Team Members
1. Aryan Mehta
2. Divyangna Sharma
3. Parth Jindal