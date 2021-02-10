import pandas as pd
from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load('models/d2v.model')
df = pd.read_csv('database/ecommerce.csv')

def similar(product_name = False, product_id = False):
    if product_name:
        productDes = eval(df.product_description_clean.loc[(df.product_title == product_name)].values[0])
    elif product_id:
        productDes = eval(df.product_description_clean.loc[(df.unit_id == product_id)].values[0])
    else:
        print("No Products")
        return
    productVec = model.infer_vector(productDes)
    similar = model.docvecs.most_similar([productVec])
    
    recommendations = []
    for tag, similarity in similar[1:]:
#         print(df.product_title.iloc[tag])
#         print(similarity)
        recommendations.append({'title':df.product_title.iloc[tag], 'similarity':similarity})
        
    return recommendations

