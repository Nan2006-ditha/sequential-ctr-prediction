from datasets import load_dataset

DATASET_NAME = "reczoo/TaobaoAd_x1"

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

print("First 10 sequence records:\n")

for i, row in enumerate(dataset):

    print(f"\n========== ROW {i} ==========")

    print("userid:", row["userid"])
    print("clk:", row["clk"])

    print("btag_his:", row["btag_his"])
    print("cate_his:", row["cate_his"])
    print("brand_his:", row["brand_his"])

    print("candidate adgroup_id:", row["adgroup_id"])
    print("candidate cate_id:", row["cate_id"])
    print("candidate brand:", row["brand"])
    print("candidate price:", row["price"])

    if i == 9:
        break