"""
This module contains utilities for the ANN course.
"""

from imports import *


def load_data(name="data/public_data.npz"):
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
    X_train_val = np.delete(X_train_val, index_to_remove, axis=0)
    y_train_val = np.delete(y_train_val, index_to_remove, axis=0)

    
    counting = pd.DataFrame(y_train_val, columns=["status"])["status"].value_counts()
    
    # Print dataset information
    dataset_info = f"The dataset without outliers contains {len(X_train_val)} images of plants, {counting[0]} healthy and {counting[1]} unhealthy."
    dataset_info += f"\nThe ratio of the healthy plants over the total is {counting[0]/len(X_train_val):.2f}."
    dataset_info += f"\nEach image has shape {X_train_val[0].shape}."
    dataset_info += f"\nThe labels encoding is: {labels}."
    print(dataset_info)

    return (
        X_train_val,
        y_train_val,
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
    show_label=False,
    name="images.pdf",
    
):
    # Calcola il numero totale di righe necessarie
    num_rows = (num_img + num_cols - 1) // num_cols

    # Crea una figura con il numero corretto di righe e colonne
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15 / 20 * num_cols, num_rows))

    # Itera attraverso il numero selezionato di immagini
    for i in range(num_img):
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axes[row_idx, col_idx]

        ax.imshow(X_train_val[i] / 255)
        if show_label:
            ax.set_title(f"{i}-{y_train_val[i][0]}")
        else:
            ax.set_title(f"{i}")
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


def visualize_dataset(dataset, title):
    plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)
    for i, samples in enumerate(iter(dataset.take(9))):
        images = samples["images"]
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


def sample_dataset_with_labels(n, X, y, aug=True, noise_std=0.1):
    """
    Randomly samples n indices from the indices of images in X.
    If aug=True, creates copies of the sampled images and corresponding labels, and returns X + copies, y + copied labels.
    If aug=False, removes the images and labels at the sampled indices and returns X with removed images and y with removed labels.

    Parameters:
    - n: Number of images to sample.
    - X: Dataset of images (numpy array).
    - y: Labels corresponding to the images.
    - aug: Boolean indicating whether to augment or reduce the number of images.

    Returns:
    - New dataset of images after the sampling operation.
    - New labels after the sampling operation.
    """

    X_1 = X[np.squeeze(y) == 1]
    y_1 = y[np.squeeze(y) == 1]

    # isolate the dataset of the class 0
    X_0 = X[np.squeeze(y) == 0]
    y_0 = y[np.squeeze(y) == 0]

    # Randomly sample n indices
    sampled_indices = np.random.choice(len(X_1), n, replace=False)

    # Copy the original dataset and labels if aug=True
    if aug:
        duplicated_images = np.copy(X_1[sampled_indices])
        duplicated_labels = np.copy(y_1[sampled_indices])

        # Add Gaussian noise to each resampled image
        for i in range(len(duplicated_images)):
            noise = np.random.normal(
                loc=0, scale=noise_std, size=duplicated_images[i].shape
            )
            duplicated_images[i] += noise

        # Concatenate the copies to the original dataset and labels
        result_dataset = np.concatenate((X_0, X_1, duplicated_images), axis=0)
        result_labels = np.concatenate((y_0, y_1, duplicated_labels), axis=0)
        # Shuffle the dataset
        indices = np.arange(len(result_dataset))
        np.random.shuffle(indices)
        result_dataset = result_dataset[indices]
        result_labels = result_labels[indices]
    else:
        # Remove the images and labels at the sampled indices if aug=False
        result_dataset = np.delete(X, sampled_indices, axis=0)
        result_labels = np.delete(y, sampled_indices, axis=0)

    return result_dataset, result_labels


# Example of usage:
# Suppose 'images' is your numpy array of images and 'labels' is your numpy array of labels
# new_images, new_labels = sample_dataset_with_labels(n=5, X=images, y=labels, aug=True, seed=42)
# Now 'new_images' contains the original images with 5 added copies, and 'new_labels' contains the original labels with 5 added copies.


def modify_image(img):
    modified_img = tf.image.random_contrast(img, 1.2, 1.6, SEED)
    modified_img = tf.image.random_brightness(modified_img, 15, SEED)
    modified_img = tf.clip_by_value(modified_img, 0, 255)
    return modified_img


def augment_modify(X, y):
    # isolate the dataset of the class 1
    X_1 = X[np.squeeze(y) == 1]
    y_1 = y[np.squeeze(y) == 1]

    # isolate the dataset of the class 0
    X_0 = X[np.squeeze(y) == 0]
    y_0 = y[np.squeeze(y) == 0]

    X_1_modified = tf.map_fn(lambda img: modify_image(img), X_1)

    # concatenate the modified images to the original ones
    X = np.concatenate((X_0, X_1, X_1_modified), axis=0)
    y = np.concatenate((y_0, y_1, y_1), axis=0)

    # shuffle the indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # shuffle the dataset
    X = X[indices]
    y = y[indices]

    return X, y


def remove_unhealthy(X, y):
    # isolate the dataset of the class 1
    X_1 = X[np.squeeze(y) == 1]
    y_1 = y[np.squeeze(y) == 1]

    # isolate the dataset of the class 0
    X_0 = X[np.squeeze(y) == 0]
    y_0 = y[np.squeeze(y) == 0]

    index_unhealthy_seems_healthy = [
        1,
        2,
        3,
        13,
        21,
        25,
        27,
        33,
        37,
        40,
        56,
        69,
        75,
        77,
        80,
        83,
        90,
        104,
        107,
        108,
        116,
        119,
        122,
        135,
        140,
        146,
        153,
        155,
        157,
        162,
        170,
        195,
        200,
        205,
        248,
        256,
        257,
        272,
        277,
        287,
        306,
        309,
        312,
        315,
        334,
        335,
        373,
        400,
        405,
        410,
        432,
        483,
        484,
        486,
        489,
        500,
        549,
        564,
        588,
        595,
        611,
        619,
        626,
        638,
        643,
        644,
        664,
        667,
        669,
        670,
        677,
        683,
        686,
        700,
        713,
        729,
        746,
        773,
        793,
        810,
        833,
        859,
        870,
        881,
        893,
        900,
        933,
        972,
        1014,
        1015,
        1019,
        1100,
        1101,
        1123,
        1153,
        1202,
        1206,
        1229,
        1234,
        1235,
        1283,
        1298,
        1304,
        1308,
        1328,
        1335,
        1351,
        1379,
        1393,
        1508,
        1513,
        1626,
        1628,
        1650,
        1798,
    ]

    X_1 = np.delete(X_1, index_unhealthy_seems_healthy, axis=0)
    y_1 = np.delete(y_1, index_unhealthy_seems_healthy, axis=0)

    # concatenate the two datasets
    X = np.concatenate((X_0, X_1))
    y = np.concatenate((y_0, y_1))

    # shuffle the indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # shuffle the dataset
    X = X[indices]
    y = y[indices]

    return X, y
