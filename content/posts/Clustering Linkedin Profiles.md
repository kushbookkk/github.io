
---
title: "NLP/NLU Series: Clustering Linkedin Profiles"
date: 2023-04-22
draft: false
---



## Overview:
{{< justify >}}

This notebook shows off how I built a simple model that leans heavily on the power of Sentence Transformers BERT to pull out lots of features. The model is pretty simple because it's based on K-means, but there's a ton of space to jazz it up and make it more complex. Basically, the algorithm I've got going here is a rock-solid starting point for the job of grouping similar LinkedIn profiles together.
{{< /justify >}}

Here's a brief rundown of the algorithm:

1. Extract BERT embeddings for sentences or textual data.
2. Concat them into single vector.
3. Use t-SNE to find optimal number of dimensions that explains the data.
4. Reduce dimensionality using PCA.
5. Find the optimal number of clusters from K-means by using distortion metric.
6. Fit reduced data to optimal number of clusters extracted from the step above.
7. Project Extracted clusters to original data.




```python
from utils import *
from modeling_utils import *
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
```

##### Utils, helper functions for visualizations:


```python
def plot_missing_data(df):

    missing_data = df.isnull().sum() / len(df) * 100
    missing_data = missing_data[missing_data != 0]
    missing_data.sort_values(ascending=False, inplace=True)

    plt.figure(figsize=(6, 6))
    sns.barplot(y=missing_data.index, x=missing_data)
    plt.title('Percentage of Missing Data by Feature')
    plt.xlabel('Percentage Missing (%)')
    plt.ylabel('Features')
    plt.show()

def visualize_normalized_histogram(df, column, top_n=100, figsize=(6, 20)):

    value_counts = df[column].value_counts().nlargest(top_n)
    value_counts_normalized = (value_counts / len(df) * 100).sort_values(
        ascending=True)  

    colors = plt.cm.get_cmap('tab20')(np.arange(top_n))[::-1]

    plt.figure(figsize=figsize)
    plt.barh(value_counts_normalized.index, value_counts_normalized.values, color=colors)
    plt.ylabel(column)
    plt.xlabel('Percentage')
    plt.title(f'Normalized Value Counts Histogram of {column} (Top {top_n})')
    plt.xticks(rotation=0)
    plt.show()

def visualize_top_15_category_histogram(data,
                                        category_column,
                                        cluster_column,
                                        top,
                                        title,
                                        width,
                                        height):
                                        
    top_n_categories = data[category_column].value_counts().nlargest(top).index.tolist()
    filtered_data = data[data[category_column].isin(top_n_categories)]

    fig, ax = plt.subplots(
        figsize=(width / 80, height / 80)) 
    sns.histplot(data=filtered_data,
                      x=category_column,
                      hue=cluster_column, 
                      multiple="stack",
                      ax=ax)

    ax.set_title(title)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_fontsize(10)
    plt.show()


def get_latest_dates(df):
    df['sort_key'] = np.where(df['date_to'] == 0, df['date_from'], df['date_to'])
    df = df.sort_values(['member_id', 'sort_key'], ascending=[True, False])
    latest_dates = df.groupby('member_id').first().reset_index()
    latest_dates = latest_dates.drop(columns=['sort_key'])

    return latest_dates
```

##### Utils, helper functions for preprocessing:


```python
def transform_experience_dates(experience):
    def transform_date_format(date_value):
        try:
            if isinstance(date_value, int) or date_value.isdigit():
                return str(date_value)  # Return the integer or numeric string as is
            else:
                date_string = str(date_value)
                date_object = datetime.strptime(date_string, "%b-%y")
                return date_object.strftime("%Y-%m")  # Format with year and month only
        except ValueError:
            return None

    def extract_year(value):
        if isinstance(value, str):
            pattern = r'\b(\d{4})\b'  # Regular expression pattern to match a four-digit year
            match = re.search(pattern, value)
            if match:
                return str(match.group(1))
        return None

    experience['transformed_date_from'] = experience['date_from'].apply(transform_date_format)
    experience['transformed_date_to'] = experience['date_to'].apply(transform_date_format)

    experience.loc[experience['transformed_date_from'].isnull(), 'transformed_date_from'] = experience.loc[
        experience['transformed_date_from'].isnull(), 'date_from'].apply(extract_year)
    experience.loc[experience['transformed_date_to'].isnull(), 'transformed_date_to'] = experience.loc[
        experience['transformed_date_to'].isnull(), 'date_to'].apply(extract_year)

    experience['transformed_date_from'] = experience['transformed_date_from'].str.replace(r'-\d{2}$', '', regex=True)
    experience['transformed_date_to'] = experience['transformed_date_to'].str.replace(r'-\d{2}$', '', regex=True)

    return experience
```

##### Utils, helper functions for modeling:


```python
def find_optimal_dimensions_tsne(data, perplexity_range):
    dims = []
    scores = []

    max_dim = min(3, data.shape[1] - 1)

    for dim in range(1, max_dim + 1):
        if dim > len(perplexity_range):
            break

        tsne = TSNE(n_components=dim)
        embeddings = tsne.fit_transform(data)

        dims.append(dim)
        scores.append(tsne.kl_divergence_)

    # Plot the KL divergence scores
    plt.plot(dims, scores, marker='o')
    plt.xlabel('Number of dimensions')
    plt.ylabel('KL Divergence Score')
    plt.title('t-SNE: KL Divergence')
    plt.show()

    optimal_dim_index = scores.index(min(scores))
    optimal_dimensions = dims[optimal_dim_index]

    return optimal_dimensions


def reduce_dimensionality_with_pca(data, components):
    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(data)
    return reduced_data


def fit_kmeans_and_evaluate(data,
                            n_clusters=4,
                            n_init=100,
                            max_iter=400,
                            init='k-means++', 
                            random_state=42):
    data_copy = data.copy()

    kmeans_model = KMeans(n_clusters=n_clusters,
                          n_init=n_init,
                          max_iter=max_iter,
                          init=init,
                          random_state=random_state)
                          
    kmeans_model.fit(data_copy)

    silhouette = silhouette_score(data_copy, kmeans_model.labels_, metric='euclidean')
    print('KMeans Scaled Silhouette Score: {}'.format(silhouette))

    labels = kmeans_model.labels_
    clusters = pd.concat([data_copy, pd.DataFrame({'cluster_scaled': labels})], axis=1)

    return clusters
```

####  Basic Employee Features:


```python
basic_features = pd.read_csv("Clean Data/basic_features.csv")
basic_features['member_id'] = basic_features['member_id'].astype(str)
basic_features.replace("none",np.NAN,inplace=True)
```

####  Employees Education:


```python
education = pd.read_csv("Clean Data/employees_education_cleaned.csv")
# transform member_id to string for ease of use
education["member_id"] = education["member_id"].astype(str)
education = education[education["member_id"].isin(basic_features["member_id"])]
```


```python
plot_missing_data(education)
```


![My Image Description](/clustering_linkedin_profiles/output_12_0.png)


```python
education.drop(["activities_and_societies","description"],axis=1,inplace=True)
education[["date_from","date_to"]] = education[["date_from","date_to"]].fillna(0)
education[["date_from","date_to"]] = education[["date_from","date_to"]].astype(int)
education[["title","subtitle"]] = education[["title","subtitle"]].fillna("none")
```


```python
# get the latest employee education obtained per member_id:
latest_education = get_latest_dates(education)

```


```python
visualize_normalized_histogram(latest_education, 'title', top_n=100)
```


    

![My Image Description](/clustering_linkedin_profiles/output_15_0.png)



```python
latest_education_drop_nan = latest_education.copy()
latest_education_drop_nan = latest_education_drop_nan[latest_education_drop_nan["subtitle"] != 'none']
```


```python
# None is causing a problem it might influence the segmentation algorithm:
visualize_normalized_histogram(latest_education_drop_nan, 'subtitle', top_n=100)
```


    

![My Image Description](/clustering_linkedin_profiles/output_17_0.png)


####  Employees Experience:


```python
experience  = pd.read_csv("Clean Data/employees_experience_cleaned.csv")
# transform member_id to string for ease of use
experience["member_id"] = experience["member_id"].astype(str)
experience = experience[experience["member_id"].isin(experience["member_id"])]
```


```python
plot_missing_data(experience)
```


    

![My Image Description](/clustering_linkedin_profiles/output_20_0.png)



```python
experience.drop(["description","location","Years","Months","duration","company_id"],
                 axis=1,inplace=True,
                 errors="ignore")
                 
experience[["date_from","date_to"]] = experience[["date_from","date_to"]].fillna(0)
experience["title"] = experience["title"].fillna("none")
experience.drop_duplicates(inplace=True)
```


```python
experience = transform_experience_dates(experience)
```


```python
experience = experience[["member_id","title","transformed_date_from","transformed_date_to"]]
experience.rename(columns={'transformed_date_from': 'date_from',
                           'transformed_date_to': 'date_to'},inplace=True)
```


```python
visualize_normalized_histogram(experience[experience["title"]!="none"], 'title', top_n=100)
```


![My Image Description](/clustering_linkedin_profiles/output_24_0.png)



```python
latest_experience = get_latest_dates(experience)
visualize_normalized_histogram(latest_experience, 'title', top_n=120)
```


    

![My Image Description](/clustering_linkedin_profiles/output_25_0.png)


####  Basic Features:


```python
basic_features.isnull().sum()
```




    member_id                   0
    title                      75
    location                    2
    industry                 1175
    summary                  1399
    recommendations_count       0
    country                     0
    connections_count           0
    experience_count            0
    latitude                    0
    longitude                   0
    months experience        1460
    number of positions      1460
    number of degrees         660
    years of educations       660
    dtype: int64



###### Remove columns with high percentage of missing values as they affect the results of cluster:


```python
plot_missing_data(basic_features)
```



![My Image Description](/clustering_linkedin_profiles/output_29_0.png)


```python
basic_features["industry"] = basic_features["industry"].fillna("other")
basic_features["title"] = basic_features["title"].fillna("other")
basic_features["location"] = basic_features["location"].fillna("unknown")
basic_features[["number of degrees","years of educations"]] = basic_features[
                                                              ["number of degrees",
                                                              "years of educations"]
                                                              ].fillna("0")
basic_features.drop(["months experience",
                     "number of positions",
                     "summary"],
                      axis=1,
                      inplace=True)
```
         



[//]: # (<div>)

[//]: # (<style scoped>)

[//]: # (    .dataframe tbody tr th:only-of-type {)

[//]: # (        vertical-align: middle;)

[//]: # (    })

[//]: # ()
[//]: # (    .dataframe tbody tr th {)

[//]: # (        vertical-align: top;)

[//]: # (    })

[//]: # ()
[//]: # (    .dataframe thead th {)

[//]: # (        text-align: right;)

[//]: # (    })

[//]: # (</style>)

[//]: # (<table border="1" class="dataframe">)

[//]: # (  <thead>)

[//]: # (    <tr style="text-align: right;">)

[//]: # (      <th></th>)

[//]: # (      <th>member_id</th>)

[//]: # (      <th>title</th>)

[//]: # (      <th>location</th>)

[//]: # (      <th>industry</th>)

[//]: # (      <th>recommendations_count</th>)

[//]: # (      <th>country</th>)

[//]: # (      <th>connections_count</th>)

[//]: # (      <th>experience_count</th>)

[//]: # (      <th>latitude</th>)

[//]: # (      <th>longitude</th>)

[//]: # (      <th>number of degrees</th>)

[//]: # (      <th>years of educations</th>)

[//]: # (    </tr>)

[//]: # (  </thead>)

[//]: # (  <tbody>)

[//]: # (    <tr>)

[//]: # (      <th>0</th>)

[//]: # (      <td>4665483</td>)

[//]: # (      <td>Ingénieur technico commercial chez Engie</td>)

[//]: # (      <td>Nanterre, Île-de-France, France</td>)

[//]: # (      <td>Banking</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>France</td>)

[//]: # (      <td>13</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>48.892423</td>)

[//]: # (      <td>2.215331</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>1</th>)

[//]: # (      <td>5222619</td>)

[//]: # (      <td>Manager at Harveys Furnishing</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>19</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>55.378051</td>)

[//]: # (      <td>-3.435973</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>0.0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>2</th>)

[//]: # (      <td>5504049</td>)

[//]: # (      <td>Sales Manager at Harveys Furnishing</td>)

[//]: # (      <td>Bridgend, Wales, United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>51.504286</td>)

[//]: # (      <td>-3.576945</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>7.0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>3</th>)

[//]: # (      <td>6704970</td>)

[//]: # (      <td>Assistant Manager at Harveys Furnishing</td>)

[//]: # (      <td>Greater Guildford Area, United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>31</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>51.236220</td>)

[//]: # (      <td>-0.570409</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>4</th>)

[//]: # (      <td>8192070</td>)

[//]: # (      <td>I Help Professionals Make Career  Business Bre...</td>)

[//]: # (      <td>Dallas, Texas, United States</td>)

[//]: # (      <td>Information Technology &amp; Services</td>)

[//]: # (      <td>26.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>16</td>)

[//]: # (      <td>32.776664</td>)

[//]: # (      <td>-96.796988</td>)

[//]: # (      <td>3.0</td>)

[//]: # (      <td>8.0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>5</th>)

[//]: # (      <td>8273835</td>)

[//]: # (      <td>Furniture Retail</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>other</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>146</td>)

[//]: # (      <td>4</td>)

[//]: # (      <td>55.378051</td>)

[//]: # (      <td>-3.435973</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>6.0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>6</th>)

[//]: # (      <td>9940377</td>)

[//]: # (      <td>Sr Research Engineer at BAMF Health</td>)

[//]: # (      <td>Grand Rapids, Michigan, United States</td>)

[//]: # (      <td>Medical Device</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>288</td>)

[//]: # (      <td>7</td>)

[//]: # (      <td>42.963360</td>)

[//]: # (      <td>-85.668086</td>)

[//]: # (      <td>7.0</td>)

[//]: # (      <td>15.0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>7</th>)

[//]: # (      <td>11076570</td>)

[//]: # (      <td>Head of New Business at Cube Online</td>)

[//]: # (      <td>Sydney, New South Wales, Australia</td>)

[//]: # (      <td>Events Services</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>Australia</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>10</td>)

[//]: # (      <td>-33.868820</td>)

[//]: # (      <td>151.209295</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>5.0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>8</th>)

[//]: # (      <td>15219102</td>)

[//]: # (      <td>Veneer Sales Manager at Mundy Veneer Limited</td>)

[//]: # (      <td>Taunton, England, United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>179</td>)

[//]: # (      <td>6</td>)

[//]: # (      <td>51.015344</td>)

[//]: # (      <td>-3.106849</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>4.0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>9</th>)

[//]: # (      <td>15809688</td>)

[//]: # (      <td>Senior Scientist  Computational Biology at Boe...</td>)

[//]: # (      <td>Cambridge, Massachusetts, United States</td>)

[//]: # (      <td>other</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>440</td>)

[//]: # (      <td>4</td>)

[//]: # (      <td>42.373616</td>)

[//]: # (      <td>-71.109733</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>9.0</td>)

[//]: # (    </tr>)

[//]: # (  </tbody>)

[//]: # (</table>)

[//]: # (</div>)




```python
visualize_normalized_histogram(basic_features[basic_features["title"]!="other"],
                                              "title",
                                               top_n=120)
```


    

![My Image Description](/clustering_linkedin_profiles/output_31_0.png)



```python
visualize_normalized_histogram(basic_features[basic_features["industry"]!="other"],"industry",top_n=120)
```


    


![My Image Description](/clustering_linkedin_profiles/output_32_0.png)


```python
visualize_normalized_histogram(basic_features,"location",top_n=120)
```


![My Image Description](/clustering_linkedin_profiles/output_33_0.png)


###### Merge employees basic features, the latest education and experience:


```python
latest_experience.drop(["date_from","date_to"],axis=1,inplace=True,errors="ignore")
latest_experience.rename(columns={"title":"experience_title"}, inplace=True)
latest_experience.head(10)
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
      <th>member_id</th>
      <th>experience_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000769811</td>
      <td>Sales  Channel Advisor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001027856</td>
      <td>Product Development Assistant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001731893</td>
      <td>VP RD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1002107022</td>
      <td>Digital Marketing  Ecommerce Consultant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1002900696</td>
      <td>Branch Manager</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1003503234</td>
      <td>Founding Shareholder</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1004617047</td>
      <td>RRH</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1004931912</td>
      <td>Senior Product Development Specialist</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1005561303</td>
      <td>Sales consultant</td>
    </tr>
    <tr>
      <th>9</th>
      <td>100559115</td>
      <td>Marketing B2B Specialist</td>
    </tr>
  </tbody>
</table>
</div>




```python
latest_education.drop(["date_from","date_to"],axis=1,inplace=True,errors="ignore")
latest_education.rename(columns={"title":"education_title","subtitle":"education_subtitle"}, inplace=True)
latest_education.head(10)
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
      <th>member_id</th>
      <th>education_title</th>
      <th>education_subtitle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000769811</td>
      <td>University of California Santa Barbara</td>
      <td>BA Business EconomicsPhilosophy Double Major</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001027856</td>
      <td>British Academy of Interior Design</td>
      <td>Postgraduate Diploma Interior Design</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001731893</td>
      <td>The Academic College of TelAviv Yaffo</td>
      <td>Computer Science</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1002107022</td>
      <td>Epping forest college</td>
      <td>2 A levels Computer studies</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1002900696</td>
      <td>bridge road adult education</td>
      <td>OCN Psychology criminal Psychology Psychosocia...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1003503234</td>
      <td>Lancaster University</td>
      <td>BSc Hons in Management</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1004617047</td>
      <td>Université Paris II  Assas</td>
      <td>Master II Droit et pratique des relations du t...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1004931912</td>
      <td>University of Michigan</td>
      <td>Post doc Radiopharmaceutical Chemistry in Nucl...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1005561303</td>
      <td>Rother Valley College</td>
      <td>Btec diploma in business  finance business pas...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>100559115</td>
      <td>CONMEBOL</td>
      <td>Certified Sports Managment</td>
    </tr>
  </tbody>
</table>
</div>




```python
overall_features  = basic_features.merge(latest_education, on='member_id', how='outer').merge(latest_experience, on='member_id', how='outer')
```



[//]: # ()
[//]: # (<div>)

[//]: # (<style scoped>)

[//]: # (    .dataframe tbody tr th:only-of-type {)

[//]: # (        vertical-align: middle;)

[//]: # (    })

[//]: # ()
[//]: # (    .dataframe tbody tr th {)

[//]: # (        vertical-align: top;)

[//]: # (    })

[//]: # ()
[//]: # (    .dataframe thead th {)

[//]: # (        text-align: right;)

[//]: # (    })

[//]: # (</style>)

[//]: # (<table border="1" class="dataframe">)

[//]: # (  <thead>)

[//]: # (    <tr style="text-align: right;">)

[//]: # (      <th></th>)

[//]: # (      <th>member_id</th>)

[//]: # (      <th>title</th>)

[//]: # (      <th>location</th>)

[//]: # (      <th>industry</th>)

[//]: # (      <th>recommendations_count</th>)

[//]: # (      <th>country</th>)

[//]: # (      <th>connections_count</th>)

[//]: # (      <th>experience_count</th>)

[//]: # (      <th>latitude</th>)

[//]: # (      <th>longitude</th>)

[//]: # (      <th>number of degrees</th>)

[//]: # (      <th>years of educations</th>)

[//]: # (      <th>education_title</th>)

[//]: # (      <th>education_subtitle</th>)

[//]: # (      <th>experience_title</th>)

[//]: # (    </tr>)

[//]: # (  </thead>)

[//]: # (  <tbody>)

[//]: # (    <tr>)

[//]: # (      <th>0</th>)

[//]: # (      <td>4665483</td>)

[//]: # (      <td>Ingénieur technico commercial chez Engie</td>)

[//]: # (      <td>Nanterre, Île-de-France, France</td>)

[//]: # (      <td>Banking</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>France</td>)

[//]: # (      <td>13</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>48.892423</td>)

[//]: # (      <td>2.215331</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>NaN</td>)

[//]: # (      <td>NaN</td>)

[//]: # (      <td>Ingénieur informatique</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>1</th>)

[//]: # (      <td>5222619</td>)

[//]: # (      <td>Manager at Harveys Furnishing</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>19</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>55.378051</td>)

[//]: # (      <td>-3.435973</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>University of Louisiana at Lafayette</td>)

[//]: # (      <td>none</td>)

[//]: # (      <td>Manager</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>2</th>)

[//]: # (      <td>5504049</td>)

[//]: # (      <td>Sales Manager at Harveys Furnishing</td>)

[//]: # (      <td>Bridgend, Wales, United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>51.504286</td>)

[//]: # (      <td>-3.576945</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>7.0</td>)

[//]: # (      <td>Brynteg Comprehensive School</td>)

[//]: # (      <td>none</td>)

[//]: # (      <td>Sales Manager</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>3</th>)

[//]: # (      <td>6704970</td>)

[//]: # (      <td>Assistant Manager at Harveys Furnishing</td>)

[//]: # (      <td>Greater Guildford Area, United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>31</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>51.236220</td>)

[//]: # (      <td>-0.570409</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>NaN</td>)

[//]: # (      <td>NaN</td>)

[//]: # (      <td>Assistant Manager</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>4</th>)

[//]: # (      <td>8192070</td>)

[//]: # (      <td>I Help Professionals Make Career  Business Bre...</td>)

[//]: # (      <td>Dallas, Texas, United States</td>)

[//]: # (      <td>Information Technology &amp; Services</td>)

[//]: # (      <td>26.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>16</td>)

[//]: # (      <td>32.776664</td>)

[//]: # (      <td>-96.796988</td>)

[//]: # (      <td>3.0</td>)

[//]: # (      <td>8.0</td>)

[//]: # (      <td>Harvard University</td>)

[//]: # (      <td>Bachelor of Arts BA Computer Science  Focus on...</td>)

[//]: # (      <td>SVP Customer Success</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>5</th>)

[//]: # (      <td>8273835</td>)

[//]: # (      <td>Furniture Retail</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>other</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>146</td>)

[//]: # (      <td>4</td>)

[//]: # (      <td>55.378051</td>)

[//]: # (      <td>-3.435973</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>6.0</td>)

[//]: # (      <td>Wrotham</td>)

[//]: # (      <td>Business Management Marketing and Related Supp...</td>)

[//]: # (      <td>Senior Sales</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>6</th>)

[//]: # (      <td>9940377</td>)

[//]: # (      <td>Sr Research Engineer at BAMF Health</td>)

[//]: # (      <td>Grand Rapids, Michigan, United States</td>)

[//]: # (      <td>Medical Device</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>288</td>)

[//]: # (      <td>7</td>)

[//]: # (      <td>42.963360</td>)

[//]: # (      <td>-85.668086</td>)

[//]: # (      <td>7.0</td>)

[//]: # (      <td>15.0</td>)

[//]: # (      <td>Grand Valley State University</td>)

[//]: # (      <td>Master of Science Engineering  Biomedical Engi...</td>)

[//]: # (      <td>Image Processing Research Engineer</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>7</th>)

[//]: # (      <td>11076570</td>)

[//]: # (      <td>Head of New Business at Cube Online</td>)

[//]: # (      <td>Sydney, New South Wales, Australia</td>)

[//]: # (      <td>Events Services</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>Australia</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>10</td>)

[//]: # (      <td>-33.868820</td>)

[//]: # (      <td>151.209295</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>5.0</td>)

[//]: # (      <td>University of the West of England</td>)

[//]: # (      <td>BA Hons Business Studies</td>)

[//]: # (      <td>APAC Hunter Manager</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>8</th>)

[//]: # (      <td>15219102</td>)

[//]: # (      <td>Veneer Sales Manager at Mundy Veneer Limited</td>)

[//]: # (      <td>Taunton, England, United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>179</td>)

[//]: # (      <td>6</td>)

[//]: # (      <td>51.015344</td>)

[//]: # (      <td>-3.106849</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>4.0</td>)

[//]: # (      <td>Northumbria University</td>)

[//]: # (      <td>Masters Degree MSc Hons Business with Management</td>)

[//]: # (      <td>Project Coordinator</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>9</th>)

[//]: # (      <td>15809688</td>)

[//]: # (      <td>Senior Scientist  Computational Biology at Boe...</td>)

[//]: # (      <td>Cambridge, Massachusetts, United States</td>)

[//]: # (      <td>other</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>440</td>)

[//]: # (      <td>4</td>)

[//]: # (      <td>42.373616</td>)

[//]: # (      <td>-71.109733</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>9.0</td>)

[//]: # (      <td>University of North Carolina at Chapel Hill</td>)

[//]: # (      <td>Doctor of Philosophy PhD Bioinformatics</td>)

[//]: # (      <td>Computational Biology Scientist</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>10</th>)

[//]: # (      <td>21059859</td>)

[//]: # (      <td>ML  Genomics  Datadriven biology</td>)

[//]: # (      <td>Cambridge, Massachusetts, United States</td>)

[//]: # (      <td>Computer Software</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>45</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>42.373616</td>)

[//]: # (      <td>-71.109733</td>)

[//]: # (      <td>4.0</td>)

[//]: # (      <td>16.0</td>)

[//]: # (      <td>Technical University of Munich</td>)

[//]: # (      <td>Doctor of Philosophy  PhD Computational Biology</td>)

[//]: # (      <td>Computational Biologist</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>11</th>)

[//]: # (      <td>22825932</td>)

[//]: # (      <td>Assistant Manager Harveys</td>)

[//]: # (      <td>St Osyth, Essex, United Kingdom</td>)

[//]: # (      <td>Retail</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>218</td>)

[//]: # (      <td>6</td>)

[//]: # (      <td>51.799152</td>)

[//]: # (      <td>1.075842</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>3.0</td>)

[//]: # (      <td>Sallynoggin University Dublin Ireland</td>)

[//]: # (      <td>Bachelors degree Leisure and fitness managemen...</td>)

[//]: # (      <td>Online Sales Manager</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>12</th>)

[//]: # (      <td>23378337</td>)

[//]: # (      <td>Industrial IoT Solutions Consultant</td>)

[//]: # (      <td>Denver, Colorado, United States</td>)

[//]: # (      <td>Industrial Automation</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>4</td>)

[//]: # (      <td>39.739236</td>)

[//]: # (      <td>-104.990251</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>4.0</td>)

[//]: # (      <td>LeTourneau University</td>)

[//]: # (      <td>Bachelor of Science BS Electrical and Computer...</td>)

[//]: # (      <td>Lead Solutions Engineer</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>13</th>)

[//]: # (      <td>24620931</td>)

[//]: # (      <td>ExGeneral Manager at AHFFABB</td>)

[//]: # (      <td>London, England, United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>93</td>)

[//]: # (      <td>2</td>)

[//]: # (      <td>51.507218</td>)

[//]: # (      <td>-0.127586</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>Harrow College</td>)

[//]: # (      <td>none</td>)

[//]: # (      <td>Store Manager</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>14</th>)

[//]: # (      <td>25021590</td>)

[//]: # (      <td>Senior Director Therapeutic Area Expansion at ...</td>)

[//]: # (      <td>Boston, Massachusetts, United States</td>)

[//]: # (      <td>Biotechnology</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>15</td>)

[//]: # (      <td>42.360082</td>)

[//]: # (      <td>-71.058880</td>)

[//]: # (      <td>5.0</td>)

[//]: # (      <td>16.0</td>)

[//]: # (      <td>Wharton Executive Education</td>)

[//]: # (      <td>Certificate Executive Presence and Influence P...</td>)

[//]: # (      <td>Director Head of External Research Collaborati...</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>15</th>)

[//]: # (      <td>27382998</td>)

[//]: # (      <td>Physician scientist working at interesection o...</td>)

[//]: # (      <td>Greater Chicago Area</td>)

[//]: # (      <td>Pharmaceuticals</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>12</td>)

[//]: # (      <td>41.743507</td>)

[//]: # (      <td>-88.011847</td>)

[//]: # (      <td>6.0</td>)

[//]: # (      <td>27.0</td>)

[//]: # (      <td>Yale University School of Medicine</td>)

[//]: # (      <td>MD Medicine</td>)

[//]: # (      <td>Senior Vice President and Head of Strategy and...</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>16</th>)

[//]: # (      <td>27540066</td>)

[//]: # (      <td>Business manager at ScS  Sofa Carpet Specialist</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>Retail</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>395</td>)

[//]: # (      <td>8</td>)

[//]: # (      <td>55.378051</td>)

[//]: # (      <td>-3.435973</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>University of Leeds</td>)

[//]: # (      <td>Business Management Marketing and Related Supp...</td>)

[//]: # (      <td>Branch Manager</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>17</th>)

[//]: # (      <td>27673065</td>)

[//]: # (      <td>Sr Bioinformatics Scientist NGS Technologies a...</td>)

[//]: # (      <td>Greater Boston</td>)

[//]: # (      <td>Research</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>4</td>)

[//]: # (      <td>42.360071</td>)

[//]: # (      <td>-71.058830</td>)

[//]: # (      <td>4.0</td>)

[//]: # (      <td>9.0</td>)

[//]: # (      <td>Brandeis University</td>)

[//]: # (      <td>Certificate Program Bioinformatics A</td>)

[//]: # (      <td>NGS Production Bioinformatics Data Scientist O...</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>18</th>)

[//]: # (      <td>28161267</td>)

[//]: # (      <td>Retail Professional</td>)

[//]: # (      <td>Greater Colchester Area</td>)

[//]: # (      <td>Retail</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>4</td>)

[//]: # (      <td>4</td>)

[//]: # (      <td>51.895927</td>)

[//]: # (      <td>0.891874</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>6.0</td>)

[//]: # (      <td>Brays Grove comprehensive</td>)

[//]: # (      <td>none</td>)

[//]: # (      <td>Retail Assistant</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>19</th>)

[//]: # (      <td>29195709</td>)

[//]: # (      <td>Strategic Partnerships at Stripe</td>)

[//]: # (      <td>New York, New York, United States</td>)

[//]: # (      <td>Internet</td>)

[//]: # (      <td>4.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>11</td>)

[//]: # (      <td>40.712775</td>)

[//]: # (      <td>-74.005973</td>)

[//]: # (      <td>5.0</td>)

[//]: # (      <td>4.0</td>)

[//]: # (      <td>University of San Francisco</td>)

[//]: # (      <td>Master of Science MS Global Entrepreneurship a...</td>)

[//]: # (      <td>Director Channel Partnerships</td>)

[//]: # (    </tr>)

[//]: # (  </tbody>)

[//]: # (</table>)

[//]: # (</div>)



###### additional preprocessing:


```python
string_columns = ["education_title",
                  "country",
                  "industry",
                  "location",
                  "title",
                  "education_subtitle",
                  "experience_title"]
overall_features[string_columns] = overall_features[string_columns].fillna('none')

numerical_cols = ["experience_count",
                  "connections_count",
                  "years of educations",
                  "number of degrees",
                  "recommendations_count",
                  "longitude",
                  "latitude"]
overall_features[numerical_cols] = overall_features[numerical_cols].fillna(0)
overall_features.isnull().sum()
```




    member_id                0
    title                    0
    location                 0
    industry                 0
    recommendations_count    0
    country                  0
    connections_count        0
    experience_count         0
    latitude                 0
    longitude                0
    number of degrees        0
    years of educations      0
    education_title          0
    education_subtitle       0
    experience_title         0
    dtype: int64




```python
visualize_none_percentages(overall_features)
```


    


![My Image Description](/clustering_linkedin_profiles/output_40_0.png)

###### Drop rows that contain "none" as it effects the performance of Clustering:



```python
overall_features = overall_features[~overall_features.apply(lambda row: row.astype(str).str.contains('none')).any(axis=1)]
overall_features = overall_features[~overall_features.apply(lambda row: row.astype(str).str.contains('other')).any(axis=1)]
overall_features[["title","industry","location","country","education_title","education_subtitle","experience_title"]].head(5).style.background_gradient()
```




<style type="text/css">
</style>
<table id="T_23d8a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_23d8a_level0_col0" class="col_heading level0 col0" >title</th>
      <th id="T_23d8a_level0_col1" class="col_heading level0 col1" >industry</th>
      <th id="T_23d8a_level0_col2" class="col_heading level0 col2" >location</th>
      <th id="T_23d8a_level0_col3" class="col_heading level0 col3" >country</th>
      <th id="T_23d8a_level0_col4" class="col_heading level0 col4" >education_title</th>
      <th id="T_23d8a_level0_col5" class="col_heading level0 col5" >education_subtitle</th>
      <th id="T_23d8a_level0_col6" class="col_heading level0 col6" >experience_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_23d8a_level0_row0" class="row_heading level0 row0" >4</th>
      <td id="T_23d8a_row0_col0" class="data row0 col0" >I Help Professionals Make Career  Business Breakthroughs  Coaching  Consulting  Personal Branding  Resumes  LinkedIn Profile  Thought Leadership Development</td>
      <td id="T_23d8a_row0_col1" class="data row0 col1" >Information Technology & Services</td>
      <td id="T_23d8a_row0_col2" class="data row0 col2" >Dallas, Texas, United States</td>
      <td id="T_23d8a_row0_col3" class="data row0 col3" >United States</td>
      <td id="T_23d8a_row0_col4" class="data row0 col4" >Harvard University</td>
      <td id="T_23d8a_row0_col5" class="data row0 col5" >Bachelor of Arts BA Computer Science  Focus on Artificial Technology Machine Learning and Education Techology</td>
      <td id="T_23d8a_row0_col6" class="data row0 col6" >SVP Customer Success</td>
    </tr>
    <tr>
      <th id="T_23d8a_level0_row1" class="row_heading level0 row1" >6</th>
      <td id="T_23d8a_row1_col0" class="data row1 col0" >Sr Research Engineer at BAMF Health</td>
      <td id="T_23d8a_row1_col1" class="data row1 col1" >Medical Device</td>
      <td id="T_23d8a_row1_col2" class="data row1 col2" >Grand Rapids, Michigan, United States</td>
      <td id="T_23d8a_row1_col3" class="data row1 col3" >United States</td>
      <td id="T_23d8a_row1_col4" class="data row1 col4" >Grand Valley State University</td>
      <td id="T_23d8a_row1_col5" class="data row1 col5" >Master of Science Engineering  Biomedical Engineering</td>
      <td id="T_23d8a_row1_col6" class="data row1 col6" >Image Processing Research Engineer</td>
    </tr>
    <tr>
      <th id="T_23d8a_level0_row2" class="row_heading level0 row2" >7</th>
      <td id="T_23d8a_row2_col0" class="data row2 col0" >Head of New Business at Cube Online</td>
      <td id="T_23d8a_row2_col1" class="data row2 col1" >Events Services</td>
      <td id="T_23d8a_row2_col2" class="data row2 col2" >Sydney, New South Wales, Australia</td>
      <td id="T_23d8a_row2_col3" class="data row2 col3" >Australia</td>
      <td id="T_23d8a_row2_col4" class="data row2 col4" >University of the West of England</td>
      <td id="T_23d8a_row2_col5" class="data row2 col5" >BA Hons Business Studies</td>
      <td id="T_23d8a_row2_col6" class="data row2 col6" >APAC Hunter Manager</td>
    </tr>
    <tr>
      <th id="T_23d8a_level0_row3" class="row_heading level0 row3" >8</th>
      <td id="T_23d8a_row3_col0" class="data row3 col0" >Veneer Sales Manager at Mundy Veneer Limited</td>
      <td id="T_23d8a_row3_col1" class="data row3 col1" >Furniture</td>
      <td id="T_23d8a_row3_col2" class="data row3 col2" >Taunton, England, United Kingdom</td>
      <td id="T_23d8a_row3_col3" class="data row3 col3" >United Kingdom</td>
      <td id="T_23d8a_row3_col4" class="data row3 col4" >Northumbria University</td>
      <td id="T_23d8a_row3_col5" class="data row3 col5" >Masters Degree MSc Hons Business with Management</td>
      <td id="T_23d8a_row3_col6" class="data row3 col6" >Project Coordinator</td>
    </tr>
    <tr>
      <th id="T_23d8a_level0_row4" class="row_heading level0 row4" >10</th>
      <td id="T_23d8a_row4_col0" class="data row4 col0" >ML  Genomics  Datadriven biology</td>
      <td id="T_23d8a_row4_col1" class="data row4 col1" >Computer Software</td>
      <td id="T_23d8a_row4_col2" class="data row4 col2" >Cambridge, Massachusetts, United States</td>
      <td id="T_23d8a_row4_col3" class="data row4 col3" >United States</td>
      <td id="T_23d8a_row4_col4" class="data row4 col4" >Technical University of Munich</td>
      <td id="T_23d8a_row4_col5" class="data row4 col5" >Doctor of Philosophy  PhD Computational Biology</td>
      <td id="T_23d8a_row4_col6" class="data row4 col6" >Computational Biologist</td>
    </tr>
  </tbody>
</table>




######  Extract Sentence Embeddings:


```python
### extract high dimensional embeddings using sentence transformer BERT:
title_embeddings = get_embeddings(overall_features,'title')
industry_embeddings = get_embeddings(overall_features,'industry')
location_embeddings = get_embeddings(overall_features,'location')
country_embeddings = get_embeddings(overall_features,'country')
education_title = get_embeddings(overall_features,'education_title')
education_subtitle = get_embeddings(overall_features,'education_subtitle')
experience_title = get_embeddings(overall_features,'experience_title')
```

######  Merge with simple features:


```python
merged_embeddings = np.concatenate((
    title_embeddings,
    industry_embeddings,
    location_embeddings,
    country_embeddings,
    education_title,
    education_subtitle,
    experience_title
), axis=1)

additional_numerical_features = overall_features[[
    'recommendations_count',
    'connections_count',
    'experience_count',
    'latitude',
    'longitude',
    'member_id'
]].values

simple_features = basic_features[['recommendations_count',
                                  'connections_count',
                                  'experience_count',
                                  'latitude',
                                  'longitude', ]]

final_data = np.concatenate((merged_embeddings, additional_numerical_features), axis=1)
final_data = pd.DataFrame(final_data)


# keep a list or ordered members_ids to use later for explanations:
members_ids = final_data.iloc[:, -1].tolist()

# drop members_id, as it's not used in modeling the data, and it will 
# lead to misleading results:
final_data = final_data.drop(final_data.columns[-1], axis=1)
```

###### Find Optimal Number of Components:


```python
find_optimal_dimensions_tsne(merged_embeddings, [5,10,15,20,25,30,35,40,45,50])
```

![My Image Description](/clustering_linkedin_profiles/output_48_0.png)




    



######  Reduce Embedding Dimensionality:


```python
reduced_merged_embeddings = reduce_dimensionality_with_pca(merged_embeddings,3)
reduced_merged_embeddings = pd.DataFrame(reduced_merged_embeddings)
reduced_merged_embeddings
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.309824</td>
      <td>-0.357302</td>
      <td>0.296566</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.611905</td>
      <td>-0.678734</td>
      <td>-0.044584</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.103592</td>
      <td>0.197899</td>
      <td>0.296579</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.664791</td>
      <td>-0.113497</td>
      <td>0.114943</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.656240</td>
      <td>-0.552984</td>
      <td>0.348519</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1140</th>
      <td>0.779930</td>
      <td>-0.071203</td>
      <td>0.002138</td>
    </tr>
    <tr>
      <th>1141</th>
      <td>0.859704</td>
      <td>-0.109584</td>
      <td>-0.098612</td>
    </tr>
    <tr>
      <th>1142</th>
      <td>-0.502135</td>
      <td>0.747044</td>
      <td>0.863029</td>
    </tr>
    <tr>
      <th>1143</th>
      <td>0.561427</td>
      <td>-0.077284</td>
      <td>-0.185717</td>
    </tr>
    <tr>
      <th>1144</th>
      <td>-0.542131</td>
      <td>0.734136</td>
      <td>-0.657090</td>
    </tr>
  </tbody>
</table>
<p>1145 rows × 3 columns</p>
</div>



###### Scale Simple Numerical Features:


```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(simple_features)
scaled_data = pd.DataFrame(scaled_data)
```

######  Merge with Reduced Embedding Vectors:


```python
merged_embeddings = pd.DataFrame(merged_embeddings)
all_features = pd.concat([merged_embeddings, scaled_data], axis=1)
```

###### Find optimal number of K that describes the data with the smallest distortion score:


```python
model = KMeans(n_init=10)
visualizer = KElbowVisualizer(model, k=(2,10))
visualizer.fit(reduced_merged_embeddings)
visualizer.show()
```


![My Image Description](/clustering_linkedin_profiles/output_56_0.png)




    <Axes: title={'center': 'Distortion Score Elbow for KMeans Clustering'}, xlabel='k', ylabel='distortion score'>




```python
merged_embeddings_clusters = fit_kmeans_and_evaluate(reduced_merged_embeddings,
                                                     4,
                                                     n_init=100,
                                                     max_iter=100000,
                                                     init='k-means++',
                                                     random_state=412)
```

    KMeans Scaled Silhouette Score: 0.5201332569122314



```python
# rename extracted clusters:
merged_embeddings_clusters = merged_embeddings_clusters.rename(columns={0:"component 1",
                                                                        1:"component 2",
                                                                        2:"component 3"})
merged_embeddings_clusters["member_id"] = members_ids
```


```python
overall_features["member_id"] = overall_features["member_id"].astype(int)
merged_embeddings_clusters["member_id"] = merged_embeddings_clusters["member_id"].astype(int)
overall_results = overall_features.merge(merged_embeddings_clusters,on='member_id')
overall_results["cluster_scaled_string"] = overall_results["cluster_scaled"].astype(str)
```




[//]: # (<div>)

[//]: # (<style scoped>)

[//]: # (    .dataframe tbody tr th:only-of-type {)

[//]: # (        vertical-align: middle;)

[//]: # (    })

[//]: # ()
[//]: # (    .dataframe tbody tr th {)

[//]: # (        vertical-align: top;)

[//]: # (    })

[//]: # ()
[//]: # (    .dataframe thead th {)

[//]: # (        text-align: right;)

[//]: # (    })

[//]: # (</style>)

[//]: # (<table border="1" class="dataframe">)

[//]: # (  <thead>)

[//]: # (    <tr style="text-align: right;">)

[//]: # (      <th></th>)

[//]: # (      <th>member_id</th>)

[//]: # (      <th>title</th>)

[//]: # (      <th>location</th>)

[//]: # (      <th>industry</th>)

[//]: # (      <th>recommendations_count</th>)

[//]: # (      <th>country</th>)

[//]: # (      <th>connections_count</th>)

[//]: # (      <th>experience_count</th>)

[//]: # (      <th>latitude</th>)

[//]: # (      <th>longitude</th>)

[//]: # (      <th>number of degrees</th>)

[//]: # (      <th>years of educations</th>)

[//]: # (      <th>education_title</th>)

[//]: # (      <th>education_subtitle</th>)

[//]: # (      <th>experience_title</th>)

[//]: # (      <th>component 1</th>)

[//]: # (      <th>component 2</th>)

[//]: # (      <th>component 3</th>)

[//]: # (      <th>cluster_scaled</th>)

[//]: # (      <th>cluster_scaled_string</th>)

[//]: # (    </tr>)

[//]: # (  </thead>)

[//]: # (  <tbody>)

[//]: # (    <tr>)

[//]: # (      <th>0</th>)

[//]: # (      <td>8192070</td>)

[//]: # (      <td>I Help Professionals Make Career  Business Bre...</td>)

[//]: # (      <td>Dallas, Texas, United States</td>)

[//]: # (      <td>Information Technology &amp; Services</td>)

[//]: # (      <td>26.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>16</td>)

[//]: # (      <td>32.776664</td>)

[//]: # (      <td>-96.796988</td>)

[//]: # (      <td>3.0</td>)

[//]: # (      <td>8.0</td>)

[//]: # (      <td>Harvard University</td>)

[//]: # (      <td>Bachelor of Arts BA Computer Science  Focus on...</td>)

[//]: # (      <td>SVP Customer Success</td>)

[//]: # (      <td>-0.309824</td>)

[//]: # (      <td>-0.357302</td>)

[//]: # (      <td>0.296566</td>)

[//]: # (      <td>2</td>)

[//]: # (      <td>2</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>1</th>)

[//]: # (      <td>9940377</td>)

[//]: # (      <td>Sr Research Engineer at BAMF Health</td>)

[//]: # (      <td>Grand Rapids, Michigan, United States</td>)

[//]: # (      <td>Medical Device</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>288</td>)

[//]: # (      <td>7</td>)

[//]: # (      <td>42.963360</td>)

[//]: # (      <td>-85.668086</td>)

[//]: # (      <td>7.0</td>)

[//]: # (      <td>15.0</td>)

[//]: # (      <td>Grand Valley State University</td>)

[//]: # (      <td>Master of Science Engineering  Biomedical Engi...</td>)

[//]: # (      <td>Image Processing Research Engineer</td>)

[//]: # (      <td>-0.611905</td>)

[//]: # (      <td>-0.678734</td>)

[//]: # (      <td>-0.044584</td>)

[//]: # (      <td>2</td>)

[//]: # (      <td>2</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>2</th>)

[//]: # (      <td>11076570</td>)

[//]: # (      <td>Head of New Business at Cube Online</td>)

[//]: # (      <td>Sydney, New South Wales, Australia</td>)

[//]: # (      <td>Events Services</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>Australia</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>10</td>)

[//]: # (      <td>-33.868820</td>)

[//]: # (      <td>151.209295</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>5.0</td>)

[//]: # (      <td>University of the West of England</td>)

[//]: # (      <td>BA Hons Business Studies</td>)

[//]: # (      <td>APAC Hunter Manager</td>)

[//]: # (      <td>0.103592</td>)

[//]: # (      <td>0.197899</td>)

[//]: # (      <td>0.296579</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>1</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>3</th>)

[//]: # (      <td>15219102</td>)

[//]: # (      <td>Veneer Sales Manager at Mundy Veneer Limited</td>)

[//]: # (      <td>Taunton, England, United Kingdom</td>)

[//]: # (      <td>Furniture</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>179</td>)

[//]: # (      <td>6</td>)

[//]: # (      <td>51.015344</td>)

[//]: # (      <td>-3.106849</td>)

[//]: # (      <td>2.0</td>)

[//]: # (      <td>4.0</td>)

[//]: # (      <td>Northumbria University</td>)

[//]: # (      <td>Masters Degree MSc Hons Business with Management</td>)

[//]: # (      <td>Project Coordinator</td>)

[//]: # (      <td>0.664791</td>)

[//]: # (      <td>-0.113497</td>)

[//]: # (      <td>0.114943</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>4</th>)

[//]: # (      <td>21059859</td>)

[//]: # (      <td>ML  Genomics  Datadriven biology</td>)

[//]: # (      <td>Cambridge, Massachusetts, United States</td>)

[//]: # (      <td>Computer Software</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>45</td>)

[//]: # (      <td>1</td>)

[//]: # (      <td>42.373616</td>)

[//]: # (      <td>-71.109733</td>)

[//]: # (      <td>4.0</td>)

[//]: # (      <td>16.0</td>)

[//]: # (      <td>Technical University of Munich</td>)

[//]: # (      <td>Doctor of Philosophy  PhD Computational Biology</td>)

[//]: # (      <td>Computational Biologist</td>)

[//]: # (      <td>-0.656240</td>)

[//]: # (      <td>-0.552984</td>)

[//]: # (      <td>0.348519</td>)

[//]: # (      <td>2</td>)

[//]: # (      <td>2</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>5</th>)

[//]: # (      <td>22825932</td>)

[//]: # (      <td>Assistant Manager Harveys</td>)

[//]: # (      <td>St Osyth, Essex, United Kingdom</td>)

[//]: # (      <td>Retail</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>218</td>)

[//]: # (      <td>6</td>)

[//]: # (      <td>51.799152</td>)

[//]: # (      <td>1.075842</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>3.0</td>)

[//]: # (      <td>Sallynoggin University Dublin Ireland</td>)

[//]: # (      <td>Bachelors degree Leisure and fitness managemen...</td>)

[//]: # (      <td>Online Sales Manager</td>)

[//]: # (      <td>0.917894</td>)

[//]: # (      <td>0.006978</td>)

[//]: # (      <td>-0.070045</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>0</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>6</th>)

[//]: # (      <td>23378337</td>)

[//]: # (      <td>Industrial IoT Solutions Consultant</td>)

[//]: # (      <td>Denver, Colorado, United States</td>)

[//]: # (      <td>Industrial Automation</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>4</td>)

[//]: # (      <td>39.739236</td>)

[//]: # (      <td>-104.990251</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>4.0</td>)

[//]: # (      <td>LeTourneau University</td>)

[//]: # (      <td>Bachelor of Science BS Electrical and Computer...</td>)

[//]: # (      <td>Lead Solutions Engineer</td>)

[//]: # (      <td>-0.346251</td>)

[//]: # (      <td>-0.219408</td>)

[//]: # (      <td>0.388687</td>)

[//]: # (      <td>2</td>)

[//]: # (      <td>2</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>7</th>)

[//]: # (      <td>25021590</td>)

[//]: # (      <td>Senior Director Therapeutic Area Expansion at ...</td>)

[//]: # (      <td>Boston, Massachusetts, United States</td>)

[//]: # (      <td>Biotechnology</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>15</td>)

[//]: # (      <td>42.360082</td>)

[//]: # (      <td>-71.058880</td>)

[//]: # (      <td>5.0</td>)

[//]: # (      <td>16.0</td>)

[//]: # (      <td>Wharton Executive Education</td>)

[//]: # (      <td>Certificate Executive Presence and Influence P...</td>)

[//]: # (      <td>Director Head of External Research Collaborati...</td>)

[//]: # (      <td>-0.373945</td>)

[//]: # (      <td>-0.679826</td>)

[//]: # (      <td>-0.296576</td>)

[//]: # (      <td>2</td>)

[//]: # (      <td>2</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>8</th>)

[//]: # (      <td>27382998</td>)

[//]: # (      <td>Physician scientist working at interesection o...</td>)

[//]: # (      <td>Greater Chicago Area</td>)

[//]: # (      <td>Pharmaceuticals</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United States</td>)

[//]: # (      <td>65535</td>)

[//]: # (      <td>12</td>)

[//]: # (      <td>41.743507</td>)

[//]: # (      <td>-88.011847</td>)

[//]: # (      <td>6.0</td>)

[//]: # (      <td>27.0</td>)

[//]: # (      <td>Yale University School of Medicine</td>)

[//]: # (      <td>MD Medicine</td>)

[//]: # (      <td>Senior Vice President and Head of Strategy and...</td>)

[//]: # (      <td>-0.420359</td>)

[//]: # (      <td>-0.750556</td>)

[//]: # (      <td>-0.308198</td>)

[//]: # (      <td>2</td>)

[//]: # (      <td>2</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (      <th>9</th>)

[//]: # (      <td>27540066</td>)

[//]: # (      <td>Business manager at ScS  Sofa Carpet Specialist</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>Retail</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>United Kingdom</td>)

[//]: # (      <td>395</td>)

[//]: # (      <td>8</td>)

[//]: # (      <td>55.378051</td>)

[//]: # (      <td>-3.435973</td>)

[//]: # (      <td>1.0</td>)

[//]: # (      <td>0.0</td>)

[//]: # (      <td>University of Leeds</td>)

[//]: # (      <td>Business Management Marketing and Related Supp...</td>)

[//]: # (      <td>Branch Manager</td>)

[//]: # (      <td>0.857216</td>)

[//]: # (      <td>0.083808</td>)

[//]: # (      <td>-0.129520</td>)

[//]: # (      <td>0</td>)

[//]: # (      <td>0</td>)

[//]: # (    </tr>)

[//]: # (  </tbody>)

[//]: # (</table>)

[//]: # (</div>)




```python
visualize_top_15_category_histogram(overall_results,
                                    category_column="industry",
                                    cluster_column="cluster_scaled_string",
                                    top=80,
                                    title="Clusters Labels Relative to Industry",
                                    width=1000,
                                    height=800
                                    )
```



![My Image Description](/clustering_linkedin_profiles/output_60_0.png)


```python
visualize_top_15_category_histogram(overall_results,
                                    category_column="location",
                                    cluster_column="cluster_scaled_string"
                                    ,top=50,
                                    title="Cluster Labels Relative to Location",
                                    width=1000,
                                    height=800
                                    )
```


    


![My Image Description](/clustering_linkedin_profiles/output_61_0.png)


```python
visualize_top_15_category_histogram(overall_results,
                                    category_column="title",
                                    cluster_column="cluster_scaled_string",
                                    top=30,
                                    title="Cluster Labels Relative to title",
                                    width=1000,
                                    height=800
                                    )
```


    



![My Image Description](/clustering_linkedin_profiles/output_62_0.png)

```python
visualize_top_15_category_histogram(overall_results,
                                    category_column="country",
                                    cluster_column="cluster_scaled_string",
                                    top=30,
                                    title="Cluster Labels Relative to Country",
                                    width=1000,
                                    height=800
                                    )
```


    


![My Image Description](/clustering_linkedin_profiles/output_63_0.png)


```python
visualize_top_15_category_histogram(overall_results,
                                    category_column="education_title",
                                    cluster_column="cluster_scaled_string",
                                    top=30,
                                    title="Cluster Labels Relative to Country",
                                    width=1000,
                                    height=800
                                    )
```


    
![My Image Description](/clustering_linkedin_profiles/output_64_0.png)



```python
visualize_top_15_category_histogram(overall_results,
                                    category_column="education_subtitle",
                                    cluster_column="cluster_scaled_string",
                                    top=30,
                                    title="Cluster Labels Relative to Education",
                                    width=1000,
                                    height=800
                                    )
```


    


![My Image Description](/clustering_linkedin_profiles/output_65_0.png)


```python
visualize_top_15_category_histogram(overall_results,
                                    category_column="experience_title",
                                    cluster_column="cluster_scaled_string",
                                    top=100,
                                    title="Cluster Labels Relative to Latest Experience Title",
                                    width=1200,
                                    height=800
                                    )
```


    


![My Image Description](/clustering_linkedin_profiles/output_66_0.png)


```python
overall_results.to_csv("Clean Data/overall_results.csv",index=False)
```

### Observations:
{{< justify >}}
- By squishing together the big, complex text features, K-means did a pretty solid job. With a Distortion Score of 152 and a Silhouette Score of 0.5201, we think it's a good place to start.
- We can probably make the clusters even better if we pull in more info like recommendations, education, and job history.
- The model seems to be pretty good at putting employees from the same industries together. Take a look at cluster 2 - it grouped retail and furniture folks together.
- In terms of where people are from, the model's done a good job. It's been putting places from the same country together, like Israel and its districts and cities in cluster 0, and it did the same with US cities in cluster 1. This kind of data looks like it'd work well with hierarchy-based models like HDBSCAN.
- One thing to note is that the model likes to stick all the empty values or "none" or "other" categories together, mostly because they look alike when transformed into vectors. To steer clear of this, I decided to leave them out.
- Just a heads-up that I used K-means here, which is pretty simple and can be thrown off by outliers. In the future, it could be worth making a fancier model that's tougher against outliers.
 
 {{< /justify >}}

