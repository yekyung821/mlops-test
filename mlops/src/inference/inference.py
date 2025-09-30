import os
import sys
import glob
import pickle
import pandas as pd

sys.path.append( # /opt/mlops
    os.path.dirname(
		    os.path.dirname(
				    os.path.dirname(
						    os.path.abspath(__file__)
						)
				)
		)
)

from src.model.game_item_cf import ItemCF  # 같은 디렉토리 기준
from src.utils.utils import model_dir

class ItemCFInference:
    def __init__(self, model_name: str, latest=True):
        self.model_name = model_name
        self.model_data = self.load_model(latest)
        self.model = ItemCF()
        # sim_matrix가 numpy array로 저장되어 있으므로 DataFrame으로 변환
        self.train_matrix = self.model_data["train_matrix"]
        self.model.item_similarity_df = pd.DataFrame(
            self.model_data["sim_matrix"],
            index=self.train_matrix.columns,
            columns=self.train_matrix.columns
        )

    def load_model(self, latest=True):
        """저장된 모델 중 최신 epoch 불러오기"""
        save_path = model_dir(self.model_name)
        files = [f for f in os.listdir(save_path) if f.endswith(".pkl")]
        if not files:
            raise FileNotFoundError("저장된 모델이 없습니다.")
        files.sort()  # 이름 기준 정렬
        target_file = files[-1] if latest else files[0]
        with open(os.path.join(save_path, target_file), "rb") as f:
            model_data = pickle.load(f)
        return model_data

    def recommend(self, user_id, top_k=5):
        """추천 결과 반환"""
        if user_id not in self.train_matrix.index:
            return []  # 없는 유저
        return self.model.recommend(user_id, self.train_matrix, top_k=top_k)
