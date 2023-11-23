import pandas as pd
import numpy as np

def fuzzy_set(a, b, c, d):
    

    return lambda t: np.piecewise(t, [t <= a, np.logical_and(t > a, t <=b), np.logical_and(t > b, t <= c), np.logical_and(t > c, t <= d), t > d],
                         [0, lambda x: (x-a)/(b-a), 1, lambda x: (d-x)/(d-c), 0])

class FuzzySystem():

    def mamdani(self, x, y):
        apply_vectorized = np.vectorize(lambda f, x: f(x), otypes=[object])

        # evaluar antecedentes
        alpha = np.minimum(apply_vectorized(self.rulebase[0], x), apply_vectorized(self.rulebase[1], y))

        # funcion para evaluar el consecuente
        C = lambda z: apply_vectorized(self.rulebase[2], z)

        # funcion para obtener el supremo de las inferencias
        fuzzy_output = lambda z: np.max(np.minimum(alpha, C(z)))
        
        # defusificacion
        # COG method
        # Aproxima las integrales por el metodo del punto medio
        a = 0 # minimo valor de recomendacion
        b = 5 # maximo valor de recomendacion
        n = 200 # numero de puntos para aproximar
        r = np.linspace(a, b, n)
        znum = (b-a) / n * np.sum([t * fuzzy_output(t) for t in r])
        zden = (b-a) / n * np.sum([fuzzy_output(t) for t in r])
        if znum==0 or zden==0:
            return 0
        
        return znum/zden

    def fuzzy_rec(self, ratings, similarity):

        # input A(similarity), B(rating)
        # output C(recommendation_level)

        products = pd.merge(ratings.reset_index(), similarity.reset_index(), how='inner', on=['asin']).loc[:,["asin", "sim", "product_rating"]]

        # evaluate ratings ^ similarity in every rule
        # alpha = 0 * [len(self.rulebase)]

        products["rec_score"] = products.apply(lambda row: self.mamdani(row["product_rating"], row["sim"]), axis=1)

        return products.loc[:, ["asin", "rec_score"]].sort_values('rec_score',ascending = False)
        
        


    def __init__(self):
        self.A = dict()
        self.B = dict()
        self.C = dict()

        # B
        # X = Similarity
        # T = {poor, average, good, excellent}
        # U = [-1, 1]
        # M: 
        # M(poor) = (0, 0, 2, 3), :                     (-1,-1,-0,6, -0,4)
        # M(average) = (1.5, 3.5, 4.5, 5.5), :          (-0.7, -0.3, -0.1, 0.1)
        # M(good) = (5, 6, 7, 8), :                     (0, 0.2, 0.4, 0.6)
        # M(excellent) = (7.5, 8.5, 10, 10) :           (0.5, 0.7, 1, 1)
        self.B["poor"] = fuzzy_set(-1,-1, -0.6, -0.4)
        self.B["average"] = fuzzy_set(-0.7, -0.3, -0.1, 0.1)
        self.B["good"] = fuzzy_set(0, 0.2, 0.4, 0.6)
        self.B["excellent"] = fuzzy_set(0.4, 0.7, 1, 1)

        # A
        # Y = Rating
        # T = {poor, average, good, excellent}
        # U = [0, 5]
        # M: 
        # M(poor) = (0, 0, 1.5, 2.5)
        # M(average) = (2, 2.5, 3, 3.5)
        # M(good) = (3, 3.5, 4, 4.5)
        # M(excellent) = (3.5, 4.5, 5, 5)
        self.A["poor"] = fuzzy_set(0, 0, 1.5, 2.5)
        self.A["average"] = fuzzy_set(2, 2.5, 3, 3.5)
        self.A["good"] = fuzzy_set(3, 3.5, 4, 4.5)
        self.A["excellent"] = fuzzy_set(3.5, 4.5, 5, 5)

        # C
        # Z = Recommendation Level
        # T = {not recommended, recommended, highly recommended}
        # U = [0, 5]
        # M: 
        # M(not recommended) = (0, 0, 1.5, 2.5)
        # M(recommended) = (2, 3, 3.5, 4.5)
        # M(highly recommended) = (3.5, 4.5, 5, 5)
        self.C["not_rec"] = fuzzy_set(0, 0, 1.5, 2.5)
        self.C["rec"] = fuzzy_set(2, 3, 3.5, 4.5)
        self.C["highly_rec"] = fuzzy_set(3.5, 4.5, 5, 5)

        self.rulebase = [
            [self.A["excellent"],self.A["excellent"], self.A["good"], self.A["average"], self.A["average"],self.A["average"], self.A["poor"]],
            [self.B["excellent"], self.B["good"],self.B["good"],self.B["excellent"],self.B["good"],self.B["average"],self.B["average"]],
            [self.C["highly_rec"], self.C["highly_rec"],self.C["highly_rec"],self.C["highly_rec"],self.C["rec"],self.C["rec"],self.C["not_rec"]]]
        

        # Rules
        # 1If rating = excellent and similarity = excellent then recommendation = highly recommended
        # 2If rating = excellent and similarity = good then recommendation = highly recommended
        # 3If rating = good and similarity = good then recommendation = highly recommended
        # 4If rating = average and similarity excellent then recommendation = hihgly recommended
        # 5If rating = average and similarity = good then recommendation = recommended
        # 6If rating = average and similarity = average then recommendation = recommended
        # 7 removed, same as 2
        # 8 removed, same as 6
        # 9 removed, same as 6
        # 10 removed, same as 6
        # 11 rating = poor and similarity = average then recommendation = not recommended
        # 12  removed, same as 2

        # As table
        # B-A-R
        #--------
        #1 E-E-HR
        #2 E-G-HR
        #3 G-G-HR
        #4 A-E-HR
        #5 A-G-R
        #6 A-A-R
        #7 P-A-NR

        # Mamdani
        # C'(z) = V[1..7](alpha ^ C(z))
        # alpha = A(similarity) ^ B(rating)


