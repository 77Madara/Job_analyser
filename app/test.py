from analysis import analyze_profile

text = """
I am a Javascript developer with experience in Typescript, React, HTML, CSS and Figma.
"""

resume = analyze_profile(text)

print(resume)