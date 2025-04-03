test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2%}')

# Generate confusion matrix
predictions = model.predict(test_generator)
plot_confusion_matrix(test_generator, predictions)