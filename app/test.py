from analysis import analyze_profile

text = """
I am a Python developer with experience in Django, FastAPI, 
TensorFlow, OpenAI, PyTorch, scikit-learn, Keras, machine learning, 
deep learning, NLP, and cloud platforms like AWS, Azure, GCP.
"""

resume = analyze_profile(text)

print(resume)