# Task 6 Error Analysis

## Objective

The objective of this task was to evaluate the trained leaf disease classification model using confusion matrix, classification report, and misclassified image analysis.

## Methodology

- Loaded the validation dataset using ImageFolder.
- Loaded the trained ResNet18 model.
- Predicted labels for validation images.
- Compared actual labels with predicted labels.
- Generated a confusion matrix.
- Generated a classification report.
- Saved misclassified images inside reports/errors.

## Classes Evaluated

['Chili___healthy', 'Chili___leaf_curl', 'Chilli__leaf_spot', 'Tomato___Bacterial_spot', 'Tomato___Late_blight', 'Tomato___healthy']

## Observations

- The confusion matrix shows correct and incorrect predictions for each class.
- The diagonal values represent correct predictions.
- The non-diagonal values represent misclassifications.
- The classification report shows precision, recall, F1-score, and support for each class.

## Error Analysis

- Some leaves may be misclassified because healthy and diseased leaves can look visually similar.
- Disease symptoms may be small, unclear, or partially visible.
- Lighting, shadows, background noise, and leaf angle can affect predictions.
- Similar disease patterns can confuse the model.

## Possible Improvements

- Add more images for weak classes.
- Use stronger data augmentation.
- Train for more epochs if needed.
- Fine-tune more ResNet18 layers.
- Use balanced class distribution.

## Conclusion

The model was evaluated using numerical metrics and visual error analysis. The confusion matrix and classification report helped understand class-wise performance, while misclassified samples helped identify practical reasons for errors.
