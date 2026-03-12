# Earth App Setup & Deployment

This guide shows how to take your project files in **globe** React/Three.js app and get it running on a fresh Ubuntu VM on GCP using **Vite** and **Nginx**.

It also includes the steps to install all app dependencies and create `vite.config.js`, so you can recreate everything from scratch on a new VM.

Your project folder on the VM will be:

```text
/home/USERNAME/globe
  App.jsx
  app.js
  app_gcp_assets.js
  index.html
  index.jsx
  package.json
  vite.config.js  (will be created later - see below)
  src/
  ...
```

Replace `USERNAME` and `YOUR_VM_IP` with your actual Linux username and VM external IP.

---

## 0. Create a new VM and open HTTP

1. In the Google Cloud Console, create a new **Ubuntu** VM (e.g. Ubuntu 22.04 LTS).
2. Make sure the VM has an **external IP**.
3. Enable HTTP access:
   - Check **“Allow HTTP traffic”** when creating the VM, **or**
   - Ensure there is a firewall rule allowing TCP port **80** to the VM.

SSH into the VM:

```bash
gcloud compute ssh YOUR_INSTANCE_NAME
# or use the web-based SSH button
```

---

## 1. Install system packages (Node.js, npm, Nginx)

On the VM:

```bash
sudo apt update
sudo apt install -y nodejs npm nginx
```

Optional: check versions:

```bash
node -v
npm -v
```

If Ubuntu’s firewall (ufw) is enabled:

```bash
sudo ufw allow 'Nginx Full'
sudo ufw reload
```

Check Nginx:

```bash
systemctl status nginx
```

At this point `http://YOUR_VM_IP/` should show the default Nginx welcome page.

---

## 2. Copy your `globe` project to the VM

On your **local machine**, from the directory that contains your `globe` folder:

```bash
scp -r ./globe USERNAME@YOUR_VM_IP:/home/USERNAME/
```

On the VM:

```bash
cd /home/USERNAME/globe
ls
```

You should see something like:

```text
App.jsx
app.js
app_gcp_assets.js
index.html
index.jsx
package.json
vite.config.js    # may or may not exist yet
src/
...
```

> If you don’t have `package.json` or `vite.config.js` yet, you’ll create them in the next step.

---

## 3. App setup: package.json, dependencies, and vite.config.js

Do all of this from the project root:

```bash
cd /home/USERNAME/globe
```

### 3.1 Create `package.json` (only if you don’t have one)

If `package.json` is missing, create it:

```bash
npm init -y
```

This creates a basic `package.json`.

### 3.2 **Install app dependencies**

These are the runtime libraries your app uses:

```bash
# Install app dependencies
npm install react react-dom react-router-dom three three-globe
```

### 3.3 **Install Vite + React plugin (dev dependencies)**

These are the build tools:

```bash
npm install -D vite @vitejs/plugin-react
```

### 3.4 **Create `vite.config.js` (or verify it)**

If `vite.config.js` doesn’t exist yet, create it now:

```bash
nano vite.config.js
```

Paste this:

```js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
});
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

If you already had a `vite.config.js`, just confirm it effectively matches this.

### 3.5 Ensure build scripts exist in `package.json`

Open `package.json`:

```bash
nano package.json
```

Make sure the `"scripts"` section includes:

```json
"scripts": {
  "dev": "vite",
  "build": "vite build",
  "preview": "vite preview"
}
```

Save and exit.

### 3.6 Build the app

Now run the production build:

```bash
npm run build
```

You should see `dist/` created with `index.html` and an `assets/` folder inside.

---

## 4. Copy the build into the Nginx web root

We’ll serve the built files from `/var/www/globe`.

```bash
# Create the web root (if it doesn't exist)
sudo mkdir -p /var/www/globe

# Clear any old deployment
sudo rm -rf /var/www/globe/*

# Copy the new build
sudo cp -r /home/USERNAME/globe/dist/* /var/www/globe/

# Give Nginx (www-data) ownership
sudo chown -R www-data:www-data /var/www/globe
```

Now `/var/www/globe` should contain:

```text
index.html
assets/
(other Vite build files)
```

---

## 5. Configure Nginx for the app (with SPA routing)

We’ll create an Nginx **server block** for this app.

### 5.1 Create the site config

```bash
sudo nano /etc/nginx/sites-available/globe
```

Paste this (replace `YOUR_VM_IP` with the VM’s external IP, or a domain later):

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name YOUR_VM_IP;

    root /var/www/globe;
    index index.html;

    # SPA routing – React Router handles client-side paths
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Optional: basic gzip
    gzip on;
    gzip_types
        text/plain
        text/css
        application/json
        application/javascript
        application/x-javascript
        text/xml
        application/xml
        application/xml+rss
        text/javascript;
}
```

Save and exit.

### 5.2 Enable the site & disable the default

```bash
# Enable your site
sudo ln -s /etc/nginx/sites-available/globe /etc/nginx/sites-enabled/globe

# Disable the default Nginx site (optional but recommended)
sudo rm -f /etc/nginx/sites-enabled/default
```

### 5.3 Test and reload Nginx

```bash
sudo nginx -t
sudo systemctl reload nginx
```

`nginx -t` should report syntax OK and test successful.

---

## 6. Verify in the browser

On your local machine, open:

```text
http://YOUR_VM_IP/
```

You should see your **globe** app (not the Nginx welcome page).
If you dont, try a hard reset (CTRL + F5) (Ctrl+Shift+R)
(forces the browser to bypass its cache and download all new content directly from the server)

Test a client-side route (React Router):

```text
http://YOUR_VM_IP/site/SomeSiteName
```

Because of the `try_files $uri $uri/ /index.html;` rule, Nginx always falls back to `index.html`, and React Router handles the route on the client.

If you still see an old page, do a **hard refresh** (e.g. `Ctrl+Shift+R` / `Cmd+Shift+R`).

---

## 7. Redeploying after code changes

When you change code in `/home/USERNAME/globe`, redeploy like this:

```bash
cd /home/USERNAME/globe

# Rebuild
npm run build

# Copy updated build to Nginx web root
sudo rm -rf /var/www/globe/*
sudo cp -r dist/* /var/www/globe/
sudo chown -R www-data:www-data /var/www/globe

# Only needed if Nginx config changed:
sudo nginx -t
sudo systemctl reload nginx
```

Your changes will be live at `http://YOUR_VM_IP/`.

---

