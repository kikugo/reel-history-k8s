// Same-origin paths: nginx (or the Vite dev proxy) forwards them to the two services.
async function call(path, options = {}) {
  const resp = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!resp.ok) throw new Error(`${path} returned ${resp.status}`);
  return resp.json();
}

export const search = (q) => call(`/api/rec/search?q=${encodeURIComponent(q)}&limit=8`);

export const recommend = (liked, k = 12) =>
  call("/api/rec/recommend", { method: "POST", body: JSON.stringify({ liked, k }) });

export const similar = (movieId, k = 6) => call(`/api/rec/similar/${movieId}?k=${k}`);

const asFilm = ({ title, year, genres }) => ({ title, year, genres });

export const explain = (liked, candidate, lang) =>
  call("/api/explain/explain", {
    method: "POST",
    body: JSON.stringify({ liked: liked.slice(-10).map(asFilm), candidate: asFilm(candidate), lang }),
  });
