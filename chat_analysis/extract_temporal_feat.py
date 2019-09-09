import csv
from utils import load_followers, parse_log
import matplotlib.pyplot as plt
from extract_feat import generate_one_hour_density_feature


if __name__ == "__main__":
    print("extract temporal feature")
    data = parse_log("chat_log_pretrain.csv")
    user_density = generate_one_hour_density_feature(data)
    # user_followers = load_followers("followers.csv")
    # X = []
    # y = []
    # for user in user_density:
    #     if user in user_followers:
    #         density = user_density[user]
    #         followers = user_followers[user]
    #         if density > 10 or followers > 10000:
    #             continue
    #         X.append(density)
    #         y.append(followers)
    # plt.scatter(X, y)
    # plt.show()
