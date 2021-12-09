## Reflection Blog Post

#### Overall, what did you achieve in your project? (Writing as a group) 

We were able to visualize a large cardiovascular disease data set, build machine learning models, use the most performant model to predict whether an individual is at risk of cardiovascular disease given his personal data, and show the result in a webapp.


#### What are two aspects of your project that you are especially proud of? 

- The accomplishment that I am proud of is achieving a roughly 73% of prediction accuracy on our test data. Though 73% of accuracy may not high enough to give one a pleasant surprise, it does show our effort compared to an accuracy of around 50% obtained by our first model. To elaborate more, we worked tirelessly to visualize the cardiovascular dataframe by plotting the heatmap, box-plot, and histogram for every feature, optimizing the data, and selecting the best combination of features. Through lots of experiments by comparing the prediction performance for each model, we eventually chose the multinomial logistic regression model trained by all features and achieved a roughly 73% of prediction accuracy on our test data.


- I am most proud of applying what we’ve learned in computer science to a real-world issue, cardiovascular disease, and presenting it on a webpage. By incorporating the machine learning model into our app.py file, we are allowed to ask users to input their physical features and then predict their risk level of cardiovascular disease. As you may know, as a math major student, the lack of application makes me feel math is somewhere tedious and powerless, but the project does provide me a brand new impression. Now, I feel like math is glamorous like so much purple heather sprung naturally out of the countryside and waited for scholars to excavate its distinct fragrance.

#### What are two things you would suggest doing to further improve your project? (You are not responsible for doing those things) 

- Although we significantly boost up the prediction accuracy to approximately 73%, there is still a long way to go. On one hand, we could do more research on the distribution for each feature, such as standard deviation, variance, covariance, etc, which enables us to clean our data more efficiently. On the other hand, the model restrictions on tree, logistic regression, and sequential may inhibit us from achieving a higher prediction accuracy. Therefore, by exploring some more appropriate models for our data, we could generate a much more precise cardiovascular prediction webpage. 

- We were not able to convert our cardiovascular webpage into a web app so that everyone can access our website easily. As a group, we did plan to achieve all these by applying Heroku, but there were lots of unexpected errors that existed. Therefore, we may commit more to Heroku and figure out the issue behind all these errors. What a pity!

#### How does what you achieved compare to what you set out to do in your proposal? (if you didn't complete everything in your proposal, that's fine!) (Writing as a group)

- We proposed to make predictions for multiple diesases but ended up focusing on cardiovascular disease

- We proposed to produce a risk index that shows the level of riskiness of suffering from a certain disease but ended up with producing a binary result indicating whether a user is at risk or not.

- We proposed to include several personalized infographics about a certain disease but ended up with giving textual information.

#### What are three things you learned from the experience of completing your project? Data analysis techniques? Python packages? Git + GitHub? Etc? 

- We learned a lot of data analysis techniques. Specifically, we know how to visualize our data distribution for each feature and then use the result of the visualization to check the outliers in our dataset. Therefore, we clean the dataset for better modeling performance.

- We learned how to incorporate our machine learning model into a webpage and build an interactive webpage interface by writing the app.py file and HTML templates. 

- We learned how to push the repository on GitHub so that people can easily access what you’ve written including Jupyter Notebook, MD file, HTML, etc. 

#### How will your experience completing this project will help you in your future studies or career? Please be as specific as possible. 

After completing my bachelor's degree at UCLA, I want to study for a master’s degree in mathematical financial engineering. As you may know, python is actually an indispensable part for analyzing the financial derivatives market. Specifically, by applying the visualization techniques, we can figure out the stock price process modeled by a geometric Brownian motion, and we also can simulate the stock price process by applying the Euler method. Indeed, python is definitely a powerful tool to visualize all these trajectories of the stock price.

Machine learning techniques enable us to build up a variety of models and train these models by a large amount of data quickly, achieving a decent prediction accuracy on unseen data, the future stock prices. 

Moreover, we can also present our model into a webpage so that everyone can access our webpage and predict the future stock price by taking different inputs, such as S&P 500 index, Dow Jones Industrial Average, Nasdaq Composite, etc. 
