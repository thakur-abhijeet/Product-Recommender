import pandas as pd
from hybrid_recommender import HybridRecommender  # Update this import if necessary

class RecommenderAPI:
    def __init__(self):
        self.model = HybridRecommender()
        # Load once during init
        self.user_df = pd.read_csv("../data/user_ratings.csv")
        self.product_df = pd.read_csv("../data/products.csv")

    def recommend_for_user(self, user_id, top_n=5):
        return self.model.get_combined_recommendations(
            user_id=user_id,
            user_df=self.user_df,
            product_df=self.product_df,
            top_n=top_n,
        )

