import tkinter as tk
from PIL import Image, ImageDraw
import serial
import time

class DigitDrawer:
    def __init__(self):
        # Define canvas and drawing characteristics
        self.root = tk.Tk()
        self.root.title("MNIST Drawer For Arduino")
        self.size = 280
        self.canvas = tk.Canvas(self.root, width=self.size, height=self.size, bg="white", cursor="cross")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.start)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.pil = Image.new("L", (self.size, self.size), 255)
        self.draw = ImageDraw.Draw(self.pil)
        self.lastx = self.lasty = None
        # Brush size selected as 28 to best replicate the MNIST dataset's number thickness
        self.brush = 28

        f = tk.Frame(self.root)
        f.pack()
        tk.Button(f, text="Clear", command=self.clear, width=10).pack(side="left", padx=5)
        tk.Button(f, text="Send", command=self.send, width=10).pack(side="left", padx=5)

        # Label to show prediction result
        self.pred = tk.Label(self.root, text="Prediction: –", font=("Arial", 18))
        self.pred.pack(pady=10)
        self.status = tk.Label(self.root, text="Draw digit (black on white)")
        self.status.pack()

        # Change port to whichever is in Arduino IDE
        self.port = "COM6"
        self.root.mainloop()

    def start(self, e):
        self.lastx, self.lasty = e.x, e.y

    def paint(self, e):
        if self.lastx is None: return
        self.canvas.create_line(self.lastx, self.lasty, e.x, e.y, fill="black", width=self.brush, capstyle=tk.ROUND, smooth=True)
        self.draw.line([self.lastx, self.lasty, e.x, e.y], fill=0, width=self.brush, joint="curve")
        self.lastx, self.lasty = e.x, e.y

    def clear(self):
        self.canvas.delete("all")
        self.pil = Image.new("L", (self.size, self.size), 255)
        self.draw = ImageDraw.Draw(self.pil)
        self.lastx = self.lasty = None
        self.pred.config(text="Prediction: –")

    def send(self):
        self.status.config(text="Sending...")
        self.root.update()

        # Resize image to 28x28 and convert to pixel values
        small = self.pil.resize((28, 28), Image.LANCZOS)
        pixels = [255 - small.getpixel((x, y)) for y in range(28) for x in range(28)]
        try:
            ser = serial.Serial(port=self.port, baudrate=115200)
            ser.reset_input_buffer()
            time.sleep(2.5)
            ser.write(bytes(pixels))
            ser.flush()
            t0 = time.time()
            pred = None
            while time.time() - t0 < 5:
                if ser.in_waiting:
                    line = ser.readline().decode(errors="ignore").strip()
                    if line.startswith("PREDICTION:"):
                        pred = line.split(":", 1)[1].strip()
                        break
                time.sleep(0.01)
            ser.close()
            if pred:
                self.pred.config(text=f"Prediction: {pred}")
                self.status.config(text="Done")
            else:
                self.status.config(text="No reply")
        except Exception as e:
            self.status.config(text=f"Error: {e}")

if __name__ == "__main__":
    DigitDrawer()