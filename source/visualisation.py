import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def correlation_heatmap(df):
    plt.figure(figsize=(10,7))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()


def rainfall_vs_yield(df):
    plt.figure(figsize=(8,6))
    plt.scatter(df["Rainfall (mm)"], df["Yield (tons/ha)"], alpha=0.7)
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Yield (tons/ha)")
    plt.title("Rainfall vs Yield")
    plt.show()


def temperature_vs_yield(df):
    plt.figure(figsize=(8,6))
    plt.scatter(df["Temperature (°C)"], df["Yield (tons/ha)"], alpha=0.7)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Yield (tons/ha)")
    plt.title("Temperature vs Yield")
    plt.show()


def average_yield(df):
    avg = df.groupby("Crop")["Yield (tons/ha)"].mean()

    plt.figure(figsize=(8,5))
    avg.plot(kind="bar")
    plt.title("Average Yield by Crop")
    plt.ylabel("Yield")
    plt.show()
