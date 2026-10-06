import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Create synthetic delivery dataset

np.random.seed(42)

delivery_times = np.random.normal(
    loc=32,
    scale=8,
    size=100
)

# Make all delivery times positive and whole numbers
delivery_times = np.round(delivery_times).astype(int)

# Add a few unusually long deliveries
delivery_times[67] = 68
delivery_times[70] = 71
delivery_times[74] = 75

data = {
    "OrderID": range(1, 101),
    "DeliveryTime(min)": delivery_times
}

df = pd.DataFrame(data)

# Save the dataset
df.to_csv("delivery_times.csv", index=False)


# 2. Load the CSV

df = pd.read_csv("delivery_times.csv")


# 3. Mean and standard deviation

mean_time = df["DeliveryTime(min)"].mean()
std_time = df["DeliveryTime(min)"].std()

print(
    f"Real delivery times -- Mean: {mean_time:.2f} min "
    f"Std: {std_time:.2f} min"
)


# 4. Simulate 100 expected delivery times

simulated_times = np.random.normal(
    loc=mean_time,
    scale=std_time,
    size=100
)

print(
    f"Simulated 100 values from Normal("
    f"mean={mean_time:.2f}, std={std_time:.2f})"
)


# 5. Missing values

missing_values = df.isnull().sum().sum()

print("Missing values:", missing_values)


# 6. Percentiles

q1 = df["DeliveryTime(min)"].quantile(0.25)
median = df["DeliveryTime(min)"].quantile(0.50)
q3 = df["DeliveryTime(min)"].quantile(0.75)

print(
    f"Percentiles -- 25th: {q1:.2f} "
    f"50th: {median:.2f} "
    f"75th: {q3:.2f}"
)


# 7. IQR outlier detection

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(
    f"IQR outlier bounds: "
    f"{lower_bound:.2f} - {upper_bound:.2f}"
)

outliers = df[
    (df["DeliveryTime(min)"] < lower_bound) |
    (df["DeliveryTime(min)"] > upper_bound)
]

print(
    "Outliers detected:",
    len(outliers),
    "deliveries",
    outliers["DeliveryTime(min)"].tolist()
)

outlier_percentage = len(outliers) / len(df) * 100

print(
    f"Percentage flagged as outliers: "
    f"{outlier_percentage:.2f}%"
)


# 8. Overlapping histograms

plt.figure(figsize=(8, 5))

plt.hist(
    df["DeliveryTime(min)"],
    bins=15,
    alpha=0.6,
    label="Real Delivery Times"
)

plt.hist(
    simulated_times,
    bins=15,
    alpha=0.6,
    label="Simulated Normal Distribution"
)

plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()

plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")
plt.close()

print("Saved delivery_comparison_histogram.png")


# 9. Boxplot

plt.figure(figsize=(6, 5))

plt.boxplot(df["DeliveryTime(min)"])

plt.ylabel("Delivery Time (minutes)")
plt.title("Delivery Time Boxplot")

plt.tight_layout()
plt.savefig("delivery_boxplot.png")
plt.close()

print("Saved delivery_boxplot.png")

# 10. Final dataset preview

print("\nFirst 10 delivery records:")
print(df.head(10))