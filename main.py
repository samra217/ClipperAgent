import yt_dlp
import matplotlib.pyplot as plt

url = "https://www.youtube.com/watch?v=zlm1aajH6gY"


opts = {
    "format": "bestvideo*+bestaudio/best",
    "outtmpl": "%(uploader)s/%(title)s.%(ext)s",
}


with yt_dlp.YoutubeDL(opts) as ydl:
    info = ydl.extract_info(url, download=True)
    print(info["title"], info["duration"])



heatmap = info.get("heatmap")
if not heatmap:
    raise SystemExit("No heatmap (video has no 'Most replayed' data)")

t = [(h["start_time"] + h["end_time"]) / 2 for h in heatmap]  # segment midpoints
v = [h["value"] for h in heatmap]
plt.plot(t,v)

plt.xlabel("Time (s)")
plt.ylabel("Relative replay intensity")
plt.show()


print(len(heatmap), info["duration"], heatmap[-1]["end_time"])
