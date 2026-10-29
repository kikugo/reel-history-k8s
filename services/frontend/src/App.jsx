import { useEffect, useState } from "react";
import { explain, recommend, search, similar } from "./api.js";

const label = (f) => (f.year ? `${f.title} (${f.year})` : f.title);

function Search({ onPick, liked }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState([]);

  useEffect(() => {
    if (q.trim().length < 2) {
      setHits([]);
      return;
    }
    const t = setTimeout(() => search(q).then((r) => setHits(r.items)).catch(() => setHits([])), 200);
    return () => clearTimeout(t);
  }, [q]);

  const taken = new Set(liked.map((f) => f.movieId));
  return (
    <div className="search">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Add a film you loved…"
        aria-label="Search films"
      />
      {hits.length > 0 && (
        <ul className="hits">
          {hits.map((f) => (
            <li key={f.movieId}>
              <button
                disabled={taken.has(f.movieId)}
                onClick={() => {
                  onPick(f);
                  setQ("");
                }}
              >
                {label(f)} <span className="muted">{f.genres.join(", ")}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Card({ film, liked, lang }) {
  const [why, setWhy] = useState(null);
  const [more, setMore] = useState(null);
  const [busy, setBusy] = useState(false);

  const askWhy = async () => {
    setBusy(true);
    try {
      setWhy(await explain(liked, film, lang));
    } catch (e) {
      setWhy({ text: e.message, source: "error" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <li className="card">
      <div className="title">{label(film)}</div>
      <div className="muted">{film.genres.join(" · ")}</div>
      <div className="actions">
        <button onClick={askWhy} disabled={busy}>
          {busy ? "Thinking…" : "Why this film?"}
        </button>
        <button onClick={() => similar(film.movieId).then((r) => setMore(r.items)).catch(() => setMore([]))}>
          Similar
        </button>
      </div>
      {why && (
        <p className="why">
          {why.text} <span className={`badge ${why.source}`}>{why.source}</span>
        </p>
      )}
      {more && <p className="muted">{more.length ? more.map(label).join(" · ") : "No similar films found."}</p>}
    </li>
  );
}

export default function App() {
  const [liked, setLiked] = useState([]);
  const [recs, setRecs] = useState(null);
  const [lang, setLang] = useState("en");
  const [error, setError] = useState(null);

  const run = async () => {
    setError(null);
    try {
      setRecs(await recommend(liked.map((f) => f.movieId)));
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <main>
      <header>
        <h1>Reel History</h1>
        <select value={lang} onChange={(e) => setLang(e.target.value)} aria-label="Explanation language">
          <option value="en">English</option>
          <option value="fr">Français</option>
        </select>
      </header>
      <p className="muted">Add films you loved, oldest first. The recommender reads them as a sequence.</p>

      <Search liked={liked} onPick={(f) => setLiked([...liked, f])} />
      <ol className="liked">
        {liked.map((f) => (
          <li key={f.movieId}>
            {label(f)}
            <button aria-label={`Remove ${f.title}`} onClick={() => setLiked(liked.filter((x) => x !== f))}>
              ×
            </button>
          </li>
        ))}
      </ol>
      <button className="primary" onClick={run}>
        {liked.length ? "Recommend" : "Show popular films"}
      </button>
      {error && <p className="error">{error}</p>}

      {recs && (
        <ul className="cards">
          {recs.items.map((f) => (
            <Card key={`${f.movieId}-${lang}`} film={f} liked={liked} lang={lang} />
          ))}
        </ul>
      )}
    </main>
  );
}
