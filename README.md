PS_1 - 
Running the code - 
Open the jupyter notebook in google colab and using T4 GPU and run all the import statements specified and the model call.
Provide image path of the image in next block.
Then run the next block ,it would give you image summary.
Run the next block where it would ask you your query ,it would give you the relevant relevant web search.
Running the next block would give you the web search links.

Folder structure -
PS_1 folder then - 
Ps_1.ipynb

Tech stack - 
Models - Open mini CPM (for image summary generation).
NLP - For generating web search query(spacy library and en_core_web_md(spacy's english model)).
Search API - for generating links using web search.

Explanation - 
First image summary is generated using CPM model.
Then image and query is taken as input from the user.
Now there can be two scenarios - 
1) The object in query remains the same ,adjective remains the same(blue towel example in problem statement)
2) Vice - Versa (Jacket and superman example in problem statement)
To tackle this we took all objects and their corrosponding adjectives pairs and did replacements between corresponding pairs
of image summary and user query such that the information missing in the user query (example-similar products,same products etc) get 
replaced by object(if object info is missing ) of image summary or adjective(if adjective info is missing) of image summary.
Which finally generates the web search query which goes into web search query.
