# Lesion Labeling Guide (CVAT)

We draw a box around **every lesion** on each leaf. These boxes are the ground
truth: the CNN learns from them, and we score both methods against them.

We use **CVAT** (Computer Vision Annotation Tool).

## Which CVAT setup

| Option | Cost | How the team works |
| --- | --- | --- |
| **A. Hosted, one free account each** (recommended) | Free | Each member labels their own batch in their own account, exports the boxes, and pushes them to GitHub |
| B. Self-hosted with Docker | Free | One shared project that everyone logs into; one member runs it |

The free hosted plan at [app.cvat.ai](https://app.cvat.ai) allows 1 project and
3 tasks per account, and team collaboration requires a paid plan. That fits
option A: one project per person, one task per batch of images.

## Option A: hosted CVAT

### 1. Create the project (once per person)

1. Sign up at <https://app.cvat.ai> and click **Projects → +**.
2. Name it `apple-lesions-<yourname>`.
3. Under **Labels**, open the **Raw** tab, delete what is there, and paste the
   contents of [`cvat_labels.json`](cvat_labels.json). Click **Done**, then **Submit**.

Using the same label file keeps everyone's label names identical.

### 2. Create a task for your batch

1. Open your project and click **+ (Create a new task)**.
2. Name it after your batch, for example `batch-03-cedar-rust`.
3. Drag in the images assigned to you from `data/raw/plantvillage_apple/`.
   Batches are listed in the GitHub issue for labeling.
4. Click **Submit & Open**, then open the job.

### 3. Draw boxes

- Pick the **Rectangle** tool (or press **N**), choose the label, and drag a
  box around the lesion. Press **N** again for the next box.
- **F** moves to the next image and **D** to the previous one. Press
  **Ctrl+S** often to save.
- To fix a box, drag its corners. To delete one, select it and press **Delete**.

### 4. Export and share

1. In the task, click **Actions → Export task dataset**.
2. Choose the **YOLO 1.1** format and **leave "Save images" off**. We only need
   the box files; everyone already has the images.
3. Unzip the export into `data/labels/<batch-name>/`, then commit and push it.

**Everyone must use the same export format (YOLO 1.1)** so the files combine.

## Option B: self-hosted CVAT (shared project)

One member runs CVAT on their computer with
[Docker Desktop](https://www.docker.com/products/docker-desktop/) installed:

```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Open <http://localhost:8080> in Google Chrome (the only browser CVAT supports)
and log in. Create one project with `cvat_labels.json`, create user accounts for
the team, and assign each person their own task.

For teammates to connect from their own laptops, set `CVAT_HOST` to the host
computer's IP address before running `docker compose up -d`. They can reach it
only while that computer is on and on the same network. See the
[CVAT installation guide](https://docs.cvat.ai/docs/administration/community/basics/installation/)
for details.

## Labels

| Label | What it looks like |
| --- | --- |
| `apple_scab` | Olive-green to dark brown spots, velvety, feathery edges |
| `black_rot` | Tan-brown center with a purple margin ("frog eye") |
| `cedar_apple_rust` | Bright yellow-orange spots, often with a reddish edge |

Healthy leaves get **no boxes**. Do not label healthy tissue, veins, or the
background.

## Rules

1. **Tight boxes.** The box touches the outer edge of the discolored area,
   including the purple or reddish rim. Do not leave extra green space.
2. **One box per lesion.** If two spots touch but you can still see two
   centers, draw two boxes. If they have fully merged into one patch, draw one.
3. **Small spots count.** Label any spot you can clearly see at normal zoom.
   Skip specks you are unsure about.
4. **Use the leaf's folder label.** A leaf from `Apple___Cedar_apple_rust`
   gets `cedar_apple_rust` boxes. If you think a lesion is a different disease,
   note the file name in the labeling issue instead of changing the label.
5. **Not sure?** Skip the image and note it in the team channel.

## Assignments and agreement

- Each member labels their assigned batch (tracked in a GitHub issue).
- A shared sample of 20 images is labeled by **everyone**, so we can measure
  how closely our boxes agree (average IoU between labelers). If agreement is
  low, we revise these rules before labeling more.
