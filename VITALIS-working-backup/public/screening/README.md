# VITALIS Screening — cleaned build

This is a standalone HTML screening interface with a simple white scientific UI.

## Run

Open `index.html` directly, or serve the folder with:

```bash
python3 -m http.server 5173
```

Then open `http://127.0.0.1:5173`.

## Backend

The page uses the existing FastAPI endpoints and defaults to:

`http://127.0.0.1:8000`

To change the backend URL, open the browser console and run:

```js
localStorage.setItem('VITALIS_API_BASE_URL', 'http://127.0.0.1:8000')
```

The working manual cardiovascular flow uses `/predict/heart`, `/explain/heart`, `/benchmark`, and `/chat`.

Report upload is intentionally UI-only until document extraction is implemented.

No prediction or benchmark values are hard-coded into the result UI.
