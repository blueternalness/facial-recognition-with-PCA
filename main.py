# cell 1 - Import Libraries
# Import required libraries


from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import numpy as np


# Allow inline display of plots in Colab
#%matplotlib inline


print("All libraries imported successfully ✓")






# cell 2 - Load LFW dataset
# Load the LFW dataset
# min_faces_per_person = 50 keeps only people with enough images
# resize = 0.4 reduces image size to speed up computation


lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)


# Extract image data
images = lfw_people.images         # shape: (n_samples, h, w)
X = lfw_people.data                # flattened data, shape: (n_samples, h*w)
y = lfw_people.target
target_names = lfw_people.target_names


n_samples, h, w = images.shape
d = X.shape[1]


print("Number of samples:", n_samples)
print("Image shape (h, w):", (h, w))
print("Flattened dimension d:", d)
print("Number of classes:", len(target_names))


# Display an example image
plt.imshow(images[0], cmap="gray")
plt.title("Example face from LFW dataset")
plt.axis("off")
plt.show()






# cell 3 - Train/Test split
# Split the dataset into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(
   X, y,
   test_size=0.25,
   random_state=42,
   stratify=y      # ensures balanced distribution of classes
)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)






# cell 4 - Train PCA
# Set the number of principal components
n_components = 150


# Create PCA model
pca = PCA(
   n_components=n_components,
   svd_solver='randomized',  # faster for high-dimensional data
   whiten=True,
   random_state=42
)


# Fit PCA on the training data
pca.fit(X_train)


# Transform both training and test sets
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


print("PCA training completed ✓")
print("Original dimension:", X_train.shape[1])
print("Reduced dimension:", X_train_pca.shape[1])






# cell 5 - Figure 1: Eigenfaces Gallery
# Function to visualize eigenfaces


def plot_eigenfaces(pca, h, w, n_row=4, n_col=4):
   """
   Display the first n_row * n_col principal components
   reshaped back into image form (Eigenfaces).
   """
   eigenfaces = pca.components_.reshape((pca.n_components_, h, w))


   plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
   for i in range(n_row * n_col):
       ax = plt.subplot(n_row, n_col, i + 1)
       ax.imshow(eigenfaces[i], cmap="gray")
       ax.set_title(f"PC {i+1}")
       ax.axis("off")


   plt.suptitle("Figure 1: Gallery of Eigenfaces")
   plt.tight_layout()
   plt.show()


# Plot the first 16 eigenfaces
plot_eigenfaces(pca, h, w, n_row=4, n_col=4)






# cell 6 - Figure 2: Screen Plot
# Explained variance ratio of each component
explained_var_ratio = pca.explained_variance_ratio_


# Cumulative explained variance
cumulative_explained = np.cumsum(explained_var_ratio)
components_axis = np.arange(1, len(cumulative_explained) + 1)


# Scree Plot
plt.figure(figsize=(6, 4))
plt.plot(components_axis, cumulative_explained, marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("Figure 2: Scree Plot")
plt.grid(True)
plt.tight_layout()
plt.show()


# Print useful values for your report
for k in [10, 20, 50, 100, len(cumulative_explained)]:
   if k <= len(cumulative_explained):
       print(f"Cumulative variance for first {k} components: {cumulative_explained[k-1]:.4f}")




