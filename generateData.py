# f8f41ca9b235468ab010e3fdfa684b85
from newsapi import NewsApiClient

MAX_DOC_LEN = 70

# Init
newsapi = NewsApiClient(api_key='f8f41ca9b235468ab010e3fdfa684b85')

def write_data(Q1, Q2):

    article_data1 = newsapi.get_everything(q=Q1, language='en', sort_by='relevancy', page_size= 100)
    article_data2 = newsapi.get_everything(q=Q2, language='en', sort_by='relevancy', page_size= 100)
    query_length = min(len(article_data1['articles']), len(article_data2['articles']))

    for topic, article_data, file_no in zip([Q1, Q2], [article_data1, article_data2], [1, 2]):
        f = open('data/topic{}.txt'.format(file_no), "w")
        for i in range(query_length):
            if i < MAX_DOC_LEN:
                f.write('{}'.format(str(article_data['articles'][i]['content'])[:-15]))
            f.write('\n')
        f.close()



print("What type of articles would you like? Type Below:")
Q1 = str(input())
print("What other type of articles would you like? Type Below:")
Q2 = str(input())

topic1 = Q1
topic2 = Q2
write_data(Q1, Q2)
print("Data Generated!")
