from matplotlib.animation import FFMpegWriter, FuncAnimation

class Visualizer:
    def __init__(self, model, save_path="animation.mp4", interval=10, fps=30, bitrate=3200):
        self.model = model
        self.save_path = save_path
        self.interval = interval
        self.fps = fps
        self.bitrate = bitrate


    def setup(self):
        raise NotImplementedError("Subclasses should implement this method.")


    def record(self, X, y, epochs=1000):
        self.X = X
        self.y = y
        self.epochs = epochs
        self.writer = FFMpegWriter(fps=self.fps, bitrate=self.bitrate)
        
        self.setup()
        
        # Create and save animation
        ani = FuncAnimation(self.fig, self.frame, frames=epochs, interval=self.interval, blit=False)
        ani.save(self.save_path, writer=self.writer)
        
        print(f"Animation saved to {self.save_path}")


    def frame(self, _):
        raise NotImplementedError("Subclasses should implement this method.")
