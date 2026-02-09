from ultralytics import YOLO

# Load the model
# Make sure best.pt is inside the 'models' folder!
model = YOLO("models/best.pt")

# Print the class names
print("------------------------------------------------")
print("CLASSES FOUND IN THIS MODEL:")
print(model.names)
print("------------------------------------------------")