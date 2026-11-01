//! CLI skeleton: build and refresh the TMDB mirror — the shape of
//! `ingest/cli.py`, wave D edition.
//!
//! ```text
//! cine-rec-ingest bootstrap --dsn postgresql://... --popularity-min 1.0
//! cine-rec-ingest refresh   --dsn postgresql://...
//! cine-rec-ingest genres    --dsn postgresql://...
//! ```
//!
//! WAVE D SCOPE: argument parsing + usage/validation parity only.
//! The command bodies need the HTTP fetcher, the export stream and
//! the bootstrap/refresh orchestration — all wave E (marked TODO
//! below). The parser is pure and unit-tested so wave E only fills
//! in `run`.

use std::fmt::Write as _;

/// The parsed command line — one variant per subcommand.
#[derive(Debug, Clone, PartialEq)]
pub struct Cli {
    pub dsn: String,
    pub api_key: String,
    pub rate: i64,
    pub command: Command,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Bootstrap {
        popularity_min: f64,
        include_adult: bool,
        limit: i64,
        skip_genres: bool,
        workers: i64,
        force: bool,
    },
    Refresh {
        days: i64,
        popularity_min: f64,
        include_adult: bool,
    },
    Genres,
}

pub const USAGE: &str = "\
usage: ingest [-h] [--dsn DSN] [--api-key API_KEY] [--rate RATE]
               {bootstrap,refresh,genres} ...

Build and refresh the TMDB mirror.

options:
  --dsn DSN
  --api-key API_KEY      max requests/second (default 40)
  --rate RATE
  -h, --help

subcommands:
  bootstrap              full load from daily exports
    --popularity-min POPULARITY_MIN   (default 0.0)
    --include-adult
    --limit LIMIT         (default 1000000000)
    --skip-genres         skip the genre-list sync (already synced)
    --workers WORKERS     (default 8)
    --force               re-fetch titles already in the DB
  refresh                 full daily loop: export diff + changes
    --days DAYS           (default 1)
    --popularity-min POPULARITY_MIN   (default 0.0)
    --include-adult
  genres                  sync the two genre lists
";

/// `main()`'s validation order: DSN first, then the API key (the
/// argparse `p.error` calls in that order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    MissingDsn,
    MissingApiKey,
    Unknown(String),
    MissingValue(String),
    BadValue { flag: String, message: String },
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::MissingDsn => write!(f, "the following arguments are required: --dsn"),
            CliError::MissingApiKey => {
                write!(f, "the following arguments are required: --api-key")
            }
            CliError::Unknown(a) => write!(f, "unrecognized argument: {a}"),
            CliError::MissingValue(flag) => write!(f, "argument {flag}: expected one value"),
            CliError::BadValue { flag, message } => write!(f, "argument {flag}: {message}"),
        }
    }
}

impl std::error::Error for CliError {}

/// Defaults mirroring `ingest/cli.py` (`--dsn`/`--api-key` default to
/// the env when the flags are absent — pass the env values in; this
/// function stays pure).
pub fn parse<I, S>(args: I, env_dsn: &str, env_api_key: &str) -> Result<Cli, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut dsn: Option<String> = None;
    let mut api_key: Option<String> = None;
    let mut rate: i64 = 40;
    let mut cmd: Option<&str> = None;
    // Subcommand flags, collected then applied per subcommand.
    let mut sub_flags: Vec<(String, Option<String>)> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--dsn" => dsn = Some(take_value(&args, &mut i, a)?),
            "--api-key" => api_key = Some(take_value(&args, &mut i, a)?),
            "--rate" => {
                let v = take_value(&args, &mut i, a)?;
                rate = v.parse().map_err(|_| CliError::BadValue {
                    flag: "--rate".into(),
                    message: format!("invalid int value: '{v}'"),
                })?;
            }
            "--help" | "-h" => return Err(CliError::Unknown("--help".into())),
            "bootstrap" | "refresh" | "genres" if cmd.is_none() => cmd = Some(a),
            other if cmd.is_some() => {
                // Subcommand flag: boolean or valued, exactly the set
                // cli.py declares.
                let valued = matches!(
                    other,
                    "--popularity-min" | "--limit" | "--workers" | "--days"
                );
                if valued {
                    let v = take_value(&args, &mut i, other)?;
                    sub_flags.push((other.to_string(), Some(v)));
                } else if matches!(other, "--include-adult" | "--skip-genres" | "--force") {
                    sub_flags.push((other.to_string(), None));
                } else {
                    return Err(CliError::Unknown(other.to_string()));
                }
            }
            other => return Err(CliError::Unknown(other.to_string())),
        }
        i += 1;
    }

    let command = match cmd {
        Some("bootstrap") => {
            let mut c = Command::Bootstrap {
                popularity_min: 0.0,
                include_adult: false,
                limit: 1_000_000_000,
                skip_genres: false,
                workers: 8,
                force: false,
            };
            for (flag, value) in sub_flags {
                match (&mut c, flag.as_str(), value) {
                    (Command::Bootstrap { popularity_min, .. }, "--popularity-min", Some(v)) => {
                        *popularity_min = v.parse().map_err(|_| CliError::BadValue {
                            flag,
                            message: format!("invalid float value: '{v}'"),
                        })?;
                    }
                    (Command::Bootstrap { limit, .. }, "--limit", Some(v)) => {
                        *limit = v.parse().map_err(|_| CliError::BadValue {
                            flag,
                            message: format!("invalid int value: '{v}'"),
                        })?;
                    }
                    (Command::Bootstrap { workers, .. }, "--workers", Some(v)) => {
                        *workers = v.parse().map_err(|_| CliError::BadValue {
                            flag,
                            message: format!("invalid int value: '{v}'"),
                        })?;
                    }
                    (Command::Bootstrap { include_adult, .. }, "--include-adult", None) => {
                        *include_adult = true;
                    }
                    (Command::Bootstrap { skip_genres, .. }, "--skip-genres", None) => {
                        *skip_genres = true;
                    }
                    (Command::Bootstrap { force, .. }, "--force", None) => *force = true,
                    (_, f, _) => return Err(CliError::Unknown(f.to_string())),
                }
            }
            c
        }
        Some("refresh") => {
            let mut c = Command::Refresh {
                days: 1,
                popularity_min: 0.0,
                include_adult: false,
            };
            for (flag, value) in sub_flags {
                match (&mut c, flag.as_str(), value) {
                    (Command::Refresh { days, .. }, "--days", Some(v)) => {
                        *days = v.parse().map_err(|_| CliError::BadValue {
                            flag,
                            message: format!("invalid int value: '{v}'"),
                        })?;
                    }
                    (Command::Refresh { popularity_min, .. }, "--popularity-min", Some(v)) => {
                        *popularity_min = v.parse().map_err(|_| CliError::BadValue {
                            flag,
                            message: format!("invalid float value: '{v}'"),
                        })?;
                    }
                    (Command::Refresh { include_adult, .. }, "--include-adult", None) => {
                        *include_adult = true;
                    }
                    (_, f, _) => return Err(CliError::Unknown(f.to_string())),
                }
            }
            c
        }
        Some("genres") => {
            if !sub_flags.is_empty() {
                return Err(CliError::Unknown(sub_flags[0].0.clone()));
            }
            Command::Genres
        }
        _ => {
            return Err(CliError::Unknown(
                "command required: bootstrap|refresh|genres".into(),
            ))
        }
    };

    // argparse p.error order: --dsn first, then --api-key.
    let dsn = dsn.unwrap_or_else(|| env_dsn.to_string());
    if dsn.is_empty() {
        return Err(CliError::MissingDsn);
    }
    let api_key = api_key.unwrap_or_else(|| env_api_key.to_string());
    if api_key.is_empty() {
        return Err(CliError::MissingApiKey);
    }
    Ok(Cli {
        dsn,
        api_key,
        rate,
        command,
    })
}

fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, CliError> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| CliError::MissingValue(flag.to_string()))
}

/// TODO(wave E): the command bodies.
/// * `genres` — /genre/movie/list + /genre/tv/list through the HTTP
///   fetcher, then the two dimension upserts.
/// * `bootstrap` — `latest_export_url` + `iter_export_ids` stream →
///   `TmdbIngest::bootstrap` (worker pool over the fetcher).
/// * `refresh` — export diff + /changes paging (the full daily loop).
pub fn run(_cli: Cli) {
    let mut msg = String::new();
    writeln!(msg, "not wired yet (wave E): {:?}", _cli.command).unwrap();
    eprint!("{msg}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bootstrap_with_defaults() {
        let cli = parse(["bootstrap", "--dsn", "pg://x", "--api-key", "k"], "", "").unwrap();
        assert_eq!(cli.rate, 40);
        assert_eq!(
            cli.command,
            Command::Bootstrap {
                popularity_min: 0.0,
                include_adult: false,
                limit: 1_000_000_000,
                skip_genres: false,
                workers: 8,
                force: false,
            }
        );
    }

    #[test]
    fn parses_all_flags_and_env_fallback() {
        let cli = parse(
            [
                "--rate",
                "25",
                "refresh",
                "--days",
                "3",
                "--popularity-min",
                "1.5",
                "--include-adult",
            ],
            "pg://env-dsn",
            "env-key",
        )
        .unwrap();
        assert_eq!(cli.dsn, "pg://env-dsn");
        assert_eq!(cli.api_key, "env-key");
        assert_eq!(cli.rate, 25);
        assert_eq!(
            cli.command,
            Command::Refresh {
                days: 3,
                popularity_min: 1.5,
                include_adult: true
            }
        );
    }

    #[test]
    fn validation_order_dsn_before_api_key() {
        let err = parse(["genres"], "", "").unwrap_err();
        assert_eq!(err, CliError::MissingDsn);
        // DSN present (flag), key missing everywhere → the key error.
        let err = parse(["genres", "--dsn", "pg://x"], "", "").unwrap_err();
        assert_eq!(err, CliError::MissingApiKey);
    }

    #[test]
    fn rejects_unknown_flags_and_bad_values() {
        assert_eq!(
            parse(
                ["bootstrap", "--dsn", "d", "--api-key", "k", "--bogus"],
                "",
                ""
            )
            .unwrap_err(),
            CliError::Unknown("--bogus".into())
        );
        assert_eq!(
            parse(
                ["refresh", "--dsn", "d", "--api-key", "k", "--rate", "x"],
                "",
                ""
            )
            .unwrap_err(),
            CliError::BadValue {
                flag: "--rate".into(),
                message: "invalid int value: 'x'".into()
            }
        );
        assert_eq!(
            parse(
                ["genres", "--dsn", "d", "--api-key", "k", "--days", "1"],
                "",
                ""
            )
            .unwrap_err(),
            CliError::Unknown("--days".into()) // genres takes no flags
        );
    }
}
