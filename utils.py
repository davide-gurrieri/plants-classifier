"""
This module contains utilities for the ANN course.
"""

from imports import *


def data_processing(name="data/public_data.npz"):
    # Load data
    dataset = np.load("data/public_data.npz", allow_pickle=True)
    X_train_val = dataset["data"]
    y_train_val = dataset["labels"]
    labels = {0: "healthy", 1: "unhealthy"}

    # convert elements of y_train_val to 0 and 1
    y_train_val = np.array([0 if label == "healthy" else 1 for label in y_train_val])

    # Expand also the labels dimension moving from (x,) to (x, 1), with x cardinality
    y_train_val = np.expand_dims(y_train_val, axis=-1)

    # remove all the items equal to shrek or trol from the dataset
    shrek = X_train_val[58]
    trol = X_train_val[338]
    index_to_remove = []
    for i, imm in enumerate(X_train_val):
        if np.allclose(imm, shrek, atol=0.1) or np.allclose(imm, trol, atol=0.1):
            index_to_remove.append(i)
    X_outliers = X_train_val[index_to_remove]
    y_outliers = y_train_val[index_to_remove]
    X_train_val_no_out = np.delete(X_train_val, index_to_remove, axis=0)
    y_train_val_no_out = np.delete(y_train_val, index_to_remove, axis=0)

    # Print dataset information
    counting_no_out = pd.DataFrame(y_train_val_no_out, columns=["status"])[
        "status"
    ].value_counts()
    counting = pd.DataFrame(y_train_val, columns=["status"])["status"].value_counts()
    dataset_info = f"The dataset without outliers contains {len(X_train_val_no_out)} images of plants, {counting_no_out[0]} healthy and {counting_no_out[1]} unhealthy."
    dataset_info += f"\nThe ratio of the healthy plants over the total is {counting_no_out[0]/len(X_train_val_no_out):.2f}."
    dataset_info += f"\nThe ratio of the healthy plants over the total considering also outliers is {counting[0]/len(X_train_val):.2f}."
    dataset_info += f"\nEach image has shape {X_train_val_no_out[0].shape}."
    dataset_info += f"\nThe labels encoding is: {labels}."
    print(dataset_info)

    return (
        X_train_val,
        y_train_val,
        X_train_val_no_out,
        y_train_val_no_out,
        labels,
        X_outliers,
        y_outliers,
        shrek,
        trol,
    )


def plot_images(
    X_train_val,
    y_train_val,
    num_img=100,
    num_cols=20,
    show=True,
    save=False,
    name="images.pdf",
):
    # Calcola il numero totale di righe necessarie
    num_rows = (num_img + num_cols - 1) // num_cols

    # Crea una figura con il numero corretto di righe e colonne
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows))

    # Itera attraverso il numero selezionato di immagini
    for i in range(num_img):
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axes[row_idx, col_idx]

        ax.imshow(X_train_val[i] / 255)
        ax.set_title(f"{i}-{y_train_val[i][0]}")
        ax.axis("off")

    # Rimuovi eventuali assi extra che non sono stati utilizzati
    for i in range(num_img, num_rows * num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout()
    # Salva l'immagine in un file PDF
    if save:
        plt.savefig(name, format="pdf")
    if not show:
        plt.close()


# TO DO
# save object e load object
