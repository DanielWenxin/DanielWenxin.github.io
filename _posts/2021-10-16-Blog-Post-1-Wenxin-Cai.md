# §1. Create a Database

First, create a database with three tables: temperatures, stations, and countries. Information on how to access country names and relate them to temperature readings is in this lecture. Rather than merging, as we did in the linked lecture, you should keep these as three separate tables in your database.
Make sure to close the database connection after you are finished constructing it.


```python
import pandas as pd
import sqlite3
from plotly.io import write_html
conn = sqlite3.connect("temperture.db") 

# create a database in current directory called temps.db
```


```python
cursor = conn.cursor()
cursor.execute(\
    """DROP TABLE 'temperatures'""")
cursor.execute(\
    """DROP TABLE 'stations'""")
cursor.execute(\
    """DROP TABLE 'Country'""")

# drop the existing table in the database to make sure that the database is empty.
```




    <sqlite3.Cursor at 0x7f8bf25ffa40>




```python
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())

# drop the existing table in the database to make sure that the database is empty.
```

    []



```python
def prepare_df(df):
    """
    one argument: 
     - df : the dataframe that we need to clean and prepare
     
    The prepare_df function is here to do 
    few cleaning steps that we'll make before incorporating this data into our database
    Specifically, 
    - Define a new column "NewID" which is the first two letter of the column "ID"
    - Define the "Temp" column which is the yearly average temperature
    - Define the Month column
    
    Return the desire dataframe that can be incorporated into our database.
    """
    df["NewID"] = df["ID"].str[0:2]  # Define a new column "NewID" which is the first two letter of the column "ID"
    df = df.set_index(keys=["ID", "Year","NewID"])
    df = df.stack()
    df = df.reset_index()
    df = df.rename(columns = {"level_3"  : "Month" , 0 : "Temp"})
    df["Month"] = df["Month"].str[5:].astype(int)   # Define the Month column by only getting the numerical value
    df["Temp"] = df["Temp"] / 100                   # Define the "Temp" column which is the yearly average temperature
    return(df)
    # Return the desire dataframe that can be incorporated into our database.
```


```python
df_iter = pd.read_csv("temperture.csv", chunksize = 100000)
# Supplying a value of chunksize will cause read_csv() to return not a data frame but an iterator
# Each of whose elements is a piece of the data with number of rows equal to chunksize

for df in df_iter:
    df = prepare_df(df)
    df.to_sql("temperatures", conn, if_exists = "append", index = False)
    
# The df.to_sql() method writes to a "temperatures" table in the database (the conn object from earlier). 
# We need to specify if_exists to ensure that we add each piece to the table, rather than overwriting them each time.
```


```python
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv"
stations = pd.read_csv(url)
stations.to_sql("stations", conn, if_exists = "replace", index = False)
# The df.to_sql() method writes to a "stations" table in the database (the conn object from earlier). 
```


```python
url = "https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv"
Country = pd.read_csv(url)
Country = Country.rename(columns = {"FIPS 10-4":"NewID", "Name" : "Country"})
# Get the dataframe Country.
# We change the column name "FIPS 10-4" to "NewID", and the column name "Name" to "Country".
```


```python
Country.to_sql("Country", conn, if_exists = "replace", index = False)
# The df.to_sql() method writes to a "Country" table in the database (the conn object from earlier).
```

    /Users/nflsxl/opt/anaconda3/lib/python3.8/site-packages/pandas/core/generic.py:2872: UserWarning:
    
    The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
    



```python
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())
# Now we can check that we have a database containing three tables: temperature, stations, Country
```

    [('temperatures',), ('stations',), ('Country',)]



```python
conn.close()   #Close the database connection after we finished constructing it.
```

# §2. Write a Query Function
Write a function called query_climate_database() which accepts four arguments:

* country, a string giving the name of a country for which data should be returned.

* year_begin and year_end, two integers giving the earliest and latest years for which should be returned.

* month, an integer giving the month of the year for which should be returned.

The return value of query_climate_database() is a Pandas dataframe of temperature readings for the specified country, in the specified date range, in the specified month of the year. This dataframe should have columns for:

* The station name.
* The latitude of the station.
* The longitude of the station.
* The name of the country in which the station is located.
* The year in which the reading was taken.
* The month in which the reading was taken.
* The average temperature at the specified station during the specified year and month.
(Note: the temperatures in the raw data are already averages by month, so you don’t have to do any aggregation at this stage.)


```python
def query_climate_database(country, year_begin, year_end, month):
    """
    For function query_climate_database, we have four imputs:
      - country, the input that specifies the country name of the dataframe to be returned.
      - year_begin, the integer that give the specific year that our dataframe should begin with.
      - year_end, the integer that give the specific year that our dataframe should end.
      - month, the that give the month that our dataframe should return.
    
    The function will return a dataframe consists of columns of 
    temp for the selected country, month, from year_begin to year_end.
    """
    cmd = f"""
    SELECT S.name,S.latitude,S.longitude,C.Country,T.year,T.month,T.Temp
    FROM temperatures T 
    LEFT JOIN stations S ON T.id = S.id
    LEFT JOIN Country C ON T.NewID = C.NewID
    WHERE T.year >= {year_begin} AND T.year <= {year_end} AND T.month = {month} AND C.Country = '{country}'
    """
    df = pd.read_sql_query(cmd, conn)
    
    return df
# return a dataframe consists of columns of temp for the selected country, month, from year_begin to year_end.
```


```python
conn = sqlite3.connect("temperture.db")
```


```python
TheData = query_climate_database("India", 1980, 2020, 1)
TheData
# The return value of query_climate_database() is a Pandas dataframe of temperature readings 
# for the specified country, in the specified date range, in the specified month of the year.
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
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



# §3. Write a Geographic Scatter Function for Yearly Temperature Increases
In this part, you will write a function to create visualizations that address the following question:

#### *How does the average yearly change in temperature vary within a given country?*

Write a function called temperature_coefficient_plot(). This function should accept five explicit arguments, and an undetermined number of keyword arguments.

* country, year_begin, year_end, and month should be as in the previous part.


* min_obs, the minimum required number of years of data for any given station. Only data for stations with at least min_obs years worth of data in the specified month should be plotted; the others should be filtered out. df.transform() plus filtering is a good way to achieve this task.


* **kwargs, additional keyword arguments passed to px.scatter_mapbox(). These can be used to control the colormap used, the mapbox style, etc.

The output of this function should be an interactive geographic scatterplot, constructed using Plotly Express, with a point for each station, such that the color of the point reflects an estimate of the yearly change in temperature during the specified month and time period at that station. A reasonable way to do this is to compute the first coefficient of a linear regression model at that station, as illustrated in these lecture notes.


### Please pay attention to the following details:
* The station name is shown when you hover over the corresponding point on the map.
* The estimates shown in the hover are rounded to a sober number of significant figures.
* The colorbar and overall plot have professional titles.
* The colorbar is centered at 0, so that the “middle” of the colorbar (white, in this case) corresponds to a coefficient of 0.

It’s not necessary for your plot to look exactly like mine, but please attend to details such as these. Feel free to be creative about these labels, as well as the choice of colors, as long as your result is polished overall.
You are free (and indeed encouraged) to define additional functions as needed.


```python
from sklearn.linear_model import LinearRegression

def coef(data_group):
    """
    For function coef, we have one imput:
      - data_group: The dataframe that have already been grouped by "NAME" column
      
    The function will return an estimate of the yearly change in Temp.
    """
    x = data_group[["Year"]]  # 2 brackets because X should be a df
    y = data_group["Temp"]    # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]

# we'll use our old friend, linear regression. 
# We'll use the statistical fact that, when regressing Temp against Year, 
# the coefficient of Year will be an estimate of the yearly change in Temp
```


```python
def MiniNum(df):
    """
    For function MiniNum, we have one input:
      - df: the dataframe that we tend to find out the number of years for each station.
    
    The function will add one column named "YearNum" to df where it shows the number of years for each station.
    """
    df["YearNum"] = df.groupby('NAME')['Year'].transform('max')-df.groupby('NAME')['Year'].transform('min')
    # df.transform() is a good way to achieve this task.
    return df
    # The function will add one column named "YearNum" to df where it shows the number of years for each station.
```


```python
from plotly import express as px
def temperature_coefficient_plot(country, year_begin, year_end, month,min_obs, **kwargs):
    """
    For function temperature_coefficient_plot, we have five imputs and 
    **kwargs, additional keyword arguments passed to px.scatter_mapbox():
    
      - country, the input that specifies the country name of the dataframe to be returned.
      - year_begin, the integer that give the specific year that our dataframe should begin with.
      - year_end, the integer that give the specific year that our dataframe should end.
      - month, the that give the month that our dataframe should return.
      - min_obs, the minimum required number of years of data for any given station. 
        Only data for stations with at least min_obs years worth of data in the specified month should be plotted
      - **kwargs, additional keyword arguments passed to px.scatter_mapbox(). 
        These can be used to control the colormap used, the mapbox style, etc.
        
    The function will return a geographic scatter function for Yearly Temperature Increases
    """
    df = query_climate_database(country, year_begin, year_end, month)
    # Get Pandas dataframe of temperature readings for the specified country, 
    # in the specified date range, in the specified month of the year.
    
    df = MiniNum(df)
    df = df[df["YearNum"] >= min_obs]
    # Only data for stations with at least min_obs years worth of data in the specified month should be plotted; 
    # the others should be filtered out.
    
    coefs = df.groupby(["NAME"]).apply(coef)
    coefs = coefs.reset_index()
    # By using function coef to get the coefficient of Year, an estimate of the yearly change in Temp
    
    df = pd.merge(df,coefs, on = ["NAME"]).dropna()  
    # merge the df with the column with yearly change in Temp by "NAME"
    
    df = df.rename(columns = {0 : "Estimated Yearly Increase(℃)"})
    # rename the column as "Estimated Yearly Increase(℃)"
    
    df["Estimated Yearly Increase(℃)"] = df["Estimated Yearly Increase(℃)"].round(4)
    # make the number in "Estimated Yearly Increase(℃)" a sober number of significant figures
    
    
    MonthName = ["January","February", 
                  "March","April",
                  "May","June",
                  "July","August",
                  "September","October",
                  "November","December"]
    # Create a list of month.
    
    fig = px.scatter_mapbox(df,
                       lat = "LATITUDE",
                       lon = "LONGITUDE",
                       hover_name = "NAME",
                       color = "Estimated Yearly Increase(℃)",
                       range_color=[-0.1,0.1],
                       opacity = 0.2,
                       height = 300, 
                       title = "Estimates of yearly increase in temperature in " + MonthName[month-1] + 
                            " for stations in " + country + ", " + " years " + f'{year_begin}' + " - " + f'{year_end}',    
                        **kwargs)

    # Plug in all these parameters into the function px.scatter_mapbox.
    
    fig.update_layout(margin={"r":20,"t":50,"l":20,"b":10})  # Set the layout of the plot.
    return(fig)  #Show the plot
    
    
```


```python
color_map = px.colors.diverging.RdGy_r   # choose a colormap

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                            min_obs = 10, 
                            zoom = 2, 
                            mapbox_style="carto-positron", 
                            color_continuous_scale=color_map)

# Plug in all these parameters into the function px.scatter_mapbox and get the graph.
write_html(fig, "temperature_coefficient.html")
```
{% include temperature_coefficient.html %}



# §4. Create Two More Interesting Figures

Create at least two more complex and interesting interactive data visualizations using the same data set. These plots must be of different types (e.g. line and bar, scatter and histogram, etc). In each case, you should construct your visualization from data obtained by querying the database that you created in §1. The code to construct each visualization should be wrapped in functions, such that a user could create visualizations for different parts of the data by calling these functions with different arguments. At least one of these plots must involve multiple facets (i.e. multiple axes, each of which shows a subset of the data).

Alongside the plots, you should clearly state a question that the plot addresses, similar to the question that we posed in §3. The questions for your two additional plots should be meaningfully different from each other and from the §3 question. You will likely want to define different query functions for extracting data for these new visualizations.

It is not necessary to create geographic plots for this part. Scatterplots, histograms, and line plots (among other choices) are all appropriate. Please make sure that they are complex, engaging, professional, and targeted to the questions you posed. In other words, push yourself! Don’t hesitate to ask your peers or talk to me if you’re having trouble coming up with questions or identifying plots that might be suitable for addressing those questions.

Once you’re done, commit and push your post to publish it to the web. Then, print the webpage as a PDF and submit it on Gradescope.

## First Plot

### Question: 
##### What's the relationship between the geographical location and the average yearly change in temperature for stations in three randomly selected countries ?


```python
def query_climate_database2(country1, country2, country3, year_begin, year_end, month):
    """
    For function query_climate_database, we have four imputs:
      - country, the input that specifies the country name of the dataframe to be returned.
      - year_begin, the integer that give the specific year that our dataframe should begin with.
      - year_end, the integer that give the specific year that our dataframe should end.
      - month, the that give the month that our dataframe should return.
    
    The function will return a dataframe consists of columns of 
    temp for the selected month, from year_begin to year_end for three selected countries.
    """
    
    cmd = f"""
    SELECT S.name,S.latitude,S.longitude,C.Country,T.year,T.month,T.Temp
    FROM temperatures T 
    LEFT JOIN stations S ON T.id = S.id
    LEFT JOIN Country C ON T.NewID = C.NewID
    WHERE T.year >= {year_begin} AND T.year <= {year_end} AND T.month = {month} AND C.Country = '{country1}'
    OR T.year >= {year_begin} AND T.year <= {year_end} AND T.month = {month} AND C.Country = '{country2}'
    OR T.year >= {year_begin} AND T.year <= {year_end} AND T.month = {month} AND C.Country = '{country3}'
    """
    df = pd.read_sql_query(cmd, conn)
    
    return df

# The function will return a dataframe consists of columns of 
# temp for the selected month, from year_begin to year_end for three selected countries.
```


```python
def ThreeDScatterplot(country1,country2,country3,year_begin,year_end,month,**kwargs):
    """
    For function ThreeDScatterplot, we have six imputs and 
    **kwargs, additional keyword arguments passed to px.scatter_3d():
    
      - country1, the input that specifies first country names of the dataframe to be returned.
      - country2, the input that specifies second country name of the dataframe to be returned.
      - country3, the input that specifies third country name of the dataframe to be returned.
      - year_begin, the integer that give the specific year that our dataframe should begin with.
      - year_end, the integer that give the specific year that our dataframe should end.
      - month, the that give the month that our dataframe should return.
      - **kwargs, additional keyword arguments passed to px.scatter_mapbox(). 
        These can be used to control the colormap used, the mapbox style, etc.
        
    The function will return a 3D scatterplot for Yearly Temperature Increases, longitude and latitude.
    """
    df = query_climate_database2(country1, country2, country3, year_begin, year_end, month)
    # Get Pandas dataframe of temperature readings for three specified country, 
    # in the specified date range, in the specified month of the year.
    
    coefs = df.groupby(["NAME"]).apply(coef)
    coefs = coefs.reset_index()
    # By using function coef to get the coefficient of Year, an estimate of the yearly change in Temp
    
    df = pd.merge(df,coefs, on = ["NAME"]).dropna()
    # merge the df with the column with yearly change in Temp by "NAME"
    
    df = df.rename(columns = {0 : "Estimated Yearly Increase(℃)"})
    # rename the column as "Estimated Yearly Increase(℃)"
    
    df["Estimated Yearly Increase(℃)"] = df["Estimated Yearly Increase(℃)"].round(4)
    # make the number in "Estimated Yearly Increase(℃)" a sober number of significant figures
    
    MonthName = ["January","February", 
                  "March","April",
                  "May","June",
                  "July","August",
                  "September","October",
                  "November","December"]
    # Create a list of month.
    
    fig = px.scatter_3d(df,
                    x = "LATITUDE",
                    y = "LONGITUDE",
                    z = "Estimated Yearly Increase(℃)",
                    color = "Country",
                    opacity = 0.5,
                    title = "3D Scatterplot of LATITUDE, LONGITUDE, Estimated Yearly Increase(℃) in " + MonthName[month-1] + 
                    " for stations in " + country1 + "," + country2 + "," + country3 + ", " + " years " + f'{year_begin}' + " - " + f'{year_end}',  
                    **kwargs)
    # Plug in all these parameters into the function px.scatter_3d.
    
    fig.update_layout(margin={"r":30,"t":30,"l":0,"b":0})   # Set the layout of the plot.
    return(fig)   # Show the graph
```


```python
fig = ThreeDScatterplot("China","Japan","India",1980,2000,2)
# Plug in all these parameters into the function px.scatter_3d.

write_html(fig, "ThreeDScatterplot.html")   
```
{% include ThreeDScatterplot.html %}



## Second Plot

### Question
#### What's the relationship between the longitude of stations and average yearly change in temperature for three selected countries each year in certain year range ? 


```python
def density_heatmap(country1,country2,country3,year_begin,year_end,month, **kwargs):
    """
    For function ThreeDScatterplot, we have six imputs and 
    **kwargs, additional keyword arguments passed to px.density_heatmap():
    
      - country1, the input that specifies first country names of the dataframe to be returned.
      - country2, the input that specifies second country name of the dataframe to be returned.
      - country3, the input that specifies third country name of the dataframe to be returned.
      - year_begin, the integer that give the specific year that our dataframe should begin with.
      - year_end, the integer that give the specific year that our dataframe should end.
      - month, the that give the month that our dataframe should return.
      - **kwargs, additional keyword arguments passed to px.scatter_mapbox(). 
        These can be used to control the colormap used, the mapbox style, etc.
        
    The function will return a heatmap density plot for Yearly Temperature Increases and longitude for each country 
    in each year.
    """
    
    df = query_climate_database2(country1, country2, country3, year_begin,year_end,month)
    # Get Pandas dataframe of temperature readings for three specified country, 
    # in the specified date range, in the specified month of the year.
    
    coefs = df.groupby(["NAME"]).apply(coef)
    coefs = coefs.reset_index()
    # By using function coef to get the coefficient of Year, an estimate of the yearly change in Temp
    
    df = pd.merge(df,coefs, on = ["NAME"]).dropna()
    # merge the df with the column with yearly change in Temp by "NAME"
        
    df = df.rename(columns = {0 : "Estimated Yearly Increase(℃)"})
    # rename the column as "Estimated Yearly Increase(℃)"
    
    df["Estimated Yearly Increase(℃)"] = df["Estimated Yearly Increase(℃)"].round(4)
    # make the number in "Estimated Yearly Increase(℃)" a sober number of significant figures
    
    MonthName = ["January","February", 
                  "March","April",
                  "May","June",
                  "July","August",
                  "September","October",
                  "November","December"]
    # Create a list of month.
    
    fig = px.density_heatmap(df, 
                         x = "LONGITUDE", 
                         y = "Estimated Yearly Increase(℃)",
                         facet_row = "Country",
                         facet_col = "Year" ,
                         nbinsx = 25,
                         nbinsy = 25,
                     title = "Density Heatmap of LONGITUDE and Estimated Yearly Increase(℃) in " + MonthName[month-1] + 
                    " for stations in " + country1 + "," + country2 + "," + country3 + ", " + " years " + f'{year_begin}' + " - " + f'{year_end}',  
                            **kwargs)
    # Plug in all these parameters into the function px.density_heatmap.
    
    fig.update_layout(margin={"r":60,"t":50,"l":60,"b":20})  # Set the layout of the plot.
    return(fig)    # Show the graph.

```


```python
fig = density_heatmap('China','Japan','India',2000,2010,2)
# Plug in all these parameters into the function px.density_heatmap.

write_html(fig, "density_heatmap.html")
```
{% include density_heatmap.html %}
