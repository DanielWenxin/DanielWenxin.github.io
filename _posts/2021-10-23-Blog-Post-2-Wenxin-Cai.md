---
layout: post
title: The Blog Post 2 For Wenxin
---


## In this blog post, I’m going to make a super cool web scraper to answer the following question:
What movie or TV shows share actors with your favorite movie or show?

### Part I: Describing the scraper


Here’s a link to my project repository:

```
https://github.com/DanielWenxin/DanielWenxin.github.io/blob/master/IMDB_scraper/IMDB_scraper/spiders/imdb_spider.py
```
Here’s how we set up the project:

```
1. <implementation of parse()>

	def parse(self, response):

		Tile = "fullcredits"

		CrewAndCast = response.url + Tile

		yield scrapy.Request(CrewAndCast, callback = self.parse_full_credits)


		This method works by first defining "Tile" as a string "fullcredits", and then we concatenate the response.url with "fullcredits".

		So that we can then navigate to the Cast & Crew page.

		Once there, the parse_full_credits(self,response) should be called, by specifying this method in the callback argument to a yielded scrapy.

2. <implementation of parse_full_credits()>

	def parse_full_credits(self, response):

		for actor_link in [a.attrib["href"] for a in response.css("td.primary_photo a")]:

			if actor_link:
				actor_link = response.urljoin(actor_link)

			yield scrapy.Request(actor_link, callback = self.parse_actor_page)


			We first create a list comprehension which creates a list of relative paths, one for each actor

			Looping over the path for each actor and name it as actor_link. Then, if the actor_link does exist, we concatenate the Cast & Crew page url with actor_link

			The yielded request should specify the method parse_actor_page(self, response) should be called when the actor’s page is reached

3. <implementation of parse_actor_page()>

	def parse_actor_page(self, response):

		actor_name = response.css("span.itemprop::text").get()

		for MOVIES in response.css("div.filmo-row"):
			movie_or_TV_name = [MOVIES.css("a::text").get()]

			yield {
					"actor" : actor_name,
					"movie_or_TV_name" : movie_or_TV_name
					}

			We first obatin the actor_name by applying css method searching on span.itemprop::text detected by developer tool

			Then obatin the movie_or_TV_name by looping over and applying css method searching on div.filmo-row detected by developer tool

			yield a dictionary with two key-value pairs, of the form {"actor" : actor_name, "movie_or_TV_name" : movie_or_TV_name}
```

### Part II: Table or Visualization


## Read the CSV file called results.csv

```python
import pandas as pd
results = pd.read_csv("results.csv")
# Read the CSV file called results.csv
# With columns for actor names and the movies and TV shows on which they worked.
```


```python
results
```




<div>
<style scoped>
		.dataframe tbody tr th:only-of-type {
				vertical-align: middle;
		}

		.dataframe tbody tr th {
				vertical-align: top;
		}

		.dataframe thead th {
				text-align: right;
		}
</style>
<table border="1" class="dataframe">
	<thead>
		<tr style="text-align: right;">
			<th></th>
			<th>actor</th>
			<th>movie_or_TV_name</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<th>0</th>
			<td>Obie Matthew</td>
			<td>The Batman</td>
		</tr>
		<tr>
			<th>1</th>
			<td>Obie Matthew</td>
			<td>RideBy</td>
		</tr>
		<tr>
			<th>2</th>
			<td>Obie Matthew</td>
			<td>Death on the Nile</td>
		</tr>
		<tr>
			<th>3</th>
			<td>Obie Matthew</td>
			<td>Morbius</td>
		</tr>
		<tr>
			<th>4</th>
			<td>Obie Matthew</td>
			<td>The Great</td>
		</tr>
		<tr>
			<th>...</th>
			<td>...</td>
			<td>...</td>
		</tr>
		<tr>
			<th>4587</th>
			<td>Ralph Fiennes</td>
			<td>Venecia 2005: Crónica de Carlos Boyero</td>
		</tr>
		<tr>
			<th>4588</th>
			<td>Ralph Fiennes</td>
			<td>Sendung ohne Namen</td>
		</tr>
		<tr>
			<th>4589</th>
			<td>Ralph Fiennes</td>
			<td>Reflections of Evil</td>
		</tr>
		<tr>
			<th>4590</th>
			<td>Ralph Fiennes</td>
			<td>Fleadh Report</td>
		</tr>
		<tr>
			<th>4591</th>
			<td>Ralph Fiennes</td>
			<td>The Movie Show</td>
		</tr>
	</tbody>
</table>
<p>4592 rows × 2 columns</p>
</div>



## Compute a sorted series


```python
Series = results["movie_or_TV_name"].value_counts()

# compute a sorted list by applying the function value_counts.
# with the top movies and TV shows that share actors with your favorite movie or TV show.
```

## Convert each column of "Series" into two separate dataframes


```python
DF1 = pd.DataFrame(data = Series.index, columns = ["movie"])
DF2 = pd.DataFrame(data = Series.values, columns = ["number of shared actors"])

# Convert each column of "Series" into two separate dataframes with names "movie" and "number of shared actors".
```

## Combine DF1 and DF2


```python
DF1["number of shared actors"] = DF2["number of shared actors"]
# Conbine DF2 into DF1.
```

## Compute a sorted list 
### with the top movies and TV shows that share actors with your favorite movie or TV show


```python
DF1
```




<div>
<style scoped>
		.dataframe tbody tr th:only-of-type {
				vertical-align: middle;
		}

		.dataframe tbody tr th {
				vertical-align: top;
		}

		.dataframe thead th {
				text-align: right;
		}
</style>
<table border="1" class="dataframe">
	<thead>
		<tr style="text-align: right;">
			<th></th>
			<th>movie</th>
			<th>number of shared actors</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<th>0</th>
			<td>No Time to Die</td>
			<td>78</td>
		</tr>
		<tr>
			<th>1</th>
			<td>Spectre</td>
			<td>16</td>
		</tr>
		<tr>
			<th>2</th>
			<td>Bond 25: Live Reveal</td>
			<td>14</td>
		</tr>
		<tr>
			<th>3</th>
			<td>EastEnders</td>
			<td>14</td>
		</tr>
		<tr>
			<th>4</th>
			<td>Hollywood Insider</td>
			<td>13</td>
		</tr>
		<tr>
			<th>...</th>
			<td>...</td>
			<td>...</td>
		</tr>
		<tr>
			<th>3153</th>
			<td>Dive to Bermuda Triangle</td>
			<td>1</td>
		</tr>
		<tr>
			<th>3154</th>
			<td>Cheeky</td>
			<td>1</td>
		</tr>
		<tr>
			<th>3155</th>
			<td>Heartlands</td>
			<td>1</td>
		</tr>
		<tr>
			<th>3156</th>
			<td>Merseybeat</td>
			<td>1</td>
		</tr>
		<tr>
			<th>3157</th>
			<td>The Movie Show</td>
			<td>1</td>
		</tr>
	</tbody>
</table>
<p>3158 rows × 2 columns</p>
</div>



## Feel free to be creative. 
### You can show a pandas data frame, a chart using matplotlib or plotly, or any other sensible display of the results.


```python
from plotly import express as px
from plotly.io import write_html
```


```python
fig = px.histogram(DF1, 
									 x = "movie",
									 y = "number of shared actors",
									 width = 600,
									 height = 300)

fig.update_layout(margin={"r":30,"t":170,"l":0,"b":0})

write_html(fig, "shared actors.html")

# Plot a histogram where the x-axis represents the name of each movie and the y-axis represents the number 
# of shared actors 
# save the figure as html
```
{% include shared actors.html %}
