//! Minimal proleptic-Gregorian date — exactly what `ingest/exports.py`
//! needs from Python's `datetime.date`: construction validation,
//! `day - timedelta(days=1)` and `%m_%d_%Y` formatting.
//!
//! Algorithms are Howard Hinnant's public-domain `days_from_civil` /
//! `civil_from_days` (the textbook serial-date pair). The range is
//! pinned to Python's `date` range (years 1..=9999); parity fixtures
//! pin the observable behavior (URL fragments, previous-day across
//! month/year/leap boundaries) against the real Python implementation.
//!
//! `date.today()` is deliberately absent — reading the wall clock is a
//! runtime edge owned by wave D/E wiring, not by the pure core.

use std::fmt;

/// Every table/view the engine or the ingest loader can touch — one
/// entry per day.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CivilDate {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

/// `ValueError` parity for invalid construction (messages mirror
/// CPython's `datetime.date` texts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidDate(pub String);

impl fmt::Display for InvalidDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidDate {}

/// Python's `MINYEAR`/`MAXYEAR`.
pub const MIN_YEAR: i32 = 1;
pub const MAX_YEAR: i32 = 9999;

impl CivilDate {
    /// Python `date(year, month, day)` validation and construction.
    pub fn new(year: i32, month: i32, day: i32) -> Result<Self, InvalidDate> {
        if !(MIN_YEAR..=MAX_YEAR).contains(&year) {
            return Err(InvalidDate(format!("year {year} is out of range")));
        }
        if !(1..=12).contains(&month) {
            return Err(InvalidDate("month must be in 1..12".into()));
        }
        let dim = days_in_month(year, month as u8);
        if !(1..=dim as i32).contains(&day) {
            return Err(InvalidDate("day is out of range for month".into()));
        }
        Ok(Self {
            year,
            month: month as u8,
            day: day as u8,
        })
    }

    /// Days since 1970-01-01 (may be negative).
    pub fn to_ordinal_days(self) -> i64 {
        days_from_civil(self.year, self.month, self.day)
    }

    /// Inverse of [`CivilDate::to_ordinal_days`]; `None` outside the
    /// Python `date` range.
    pub fn from_ordinal_days(days: i64) -> Option<Self> {
        let (y, m, d) = civil_from_days(days);
        if (MIN_YEAR..=MAX_YEAR).contains(&y) {
            Some(Self {
                year: y,
                month: m as u8,
                day: d as u8,
            })
        } else {
            None
        }
    }

    /// Python `date - timedelta(days=1)`. `None` at `0001-01-01`,
    /// where Python raises `OverflowError("date value out of range")`.
    pub fn previous_day(self) -> Option<Self> {
        Self::from_ordinal_days(self.to_ordinal_days() - 1)
    }

    /// Python `f"{day:%m_%d_%Y}"`: `%m`/`%d` zero-pad to two digits,
    /// while glibc's `%Y` prints the year with NO padding at all
    /// (year 9 formats as `9`, pinned by the parity fixtures).
    pub fn url_fragment(self) -> String {
        format!("{:02}_{:02}_{}", self.month, self.day, self.year)
    }
}

/// Proleptic-Gregorian leap rule.
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Days in `month` of `year`.
pub fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Hinnant's `days_from_civil` — serial day number of y-m-d.
fn days_from_civil(y: i32, m: u8, d: u8) -> i64 {
    let y = i64::from(if m <= 2 { y - 1 } else { y });
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = (i64::from(m) + 9) % 12; // Mar=0..Feb=11
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Hinnant's `civil_from_days` — inverse of [`days_from_civil`].
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (
        if m <= 2 { (y + 1) as i32 } else { y as i32 },
        m as u32,
        d as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_and_known_serials() {
        assert_eq!(CivilDate::new(1970, 1, 1).unwrap().to_ordinal_days(), 0);
        // 2000-03-01 is day 11017 after the epoch (checked against
        // Python: (date(2000,3,1) - date(1970,1,1)).days).
        assert_eq!(
            CivilDate::new(2000, 3, 1).unwrap().to_ordinal_days(),
            11_017
        );
        assert_eq!(CivilDate::new(1969, 12, 31).unwrap().to_ordinal_days(), -1);
    }

    #[test]
    fn roundtrips_over_the_whole_python_range() {
        // Every 97 days across ~3.65M days of the range plus all the
        // month/year boundaries around leap days.
        let mut days = Vec::new();
        let mut d = CivilDate::new(1, 1, 1).unwrap().to_ordinal_days();
        let last = CivilDate::new(9999, 12, 31).unwrap().to_ordinal_days();
        while d <= last {
            days.push(d);
            d += 97;
        }
        assert!(days.len() > 37_000);
        for day in days {
            let date = CivilDate::from_ordinal_days(day).unwrap();
            assert_eq!(date.to_ordinal_days(), day);
        }
    }

    #[test]
    fn previous_day_across_boundaries() {
        let cases = [
            ((2024, 3, 1), (2024, 2, 29)), // leap
            ((2023, 3, 1), (2023, 2, 28)), // non-leap
            ((1900, 3, 1), (1900, 2, 28)), // 1900 NOT a leap year
            ((2000, 3, 1), (2000, 2, 29)), // 2000 IS a leap year
            ((2024, 1, 1), (2023, 12, 31)),
            ((1, 1, 2), (1, 1, 1)),
        ];
        for ((y, m, dd), want) in cases {
            let got = CivilDate::new(y, m, dd).unwrap().previous_day().unwrap();
            let want = CivilDate::new(want.0, want.1, want.2).unwrap();
            assert_eq!(got, want, "{y}-{m}-{dd}");
        }
        // date(1,1,1) - 1 day overflows in Python.
        assert_eq!(CivilDate::new(1, 1, 1).unwrap().previous_day(), None);
    }

    #[test]
    fn validation_messages_match_python() {
        assert_eq!(
            CivilDate::new(0, 1, 1).unwrap_err(),
            InvalidDate("year 0 is out of range".into())
        );
        assert_eq!(
            CivilDate::new(2024, 13, 1).unwrap_err(),
            InvalidDate("month must be in 1..12".into())
        );
        assert_eq!(
            CivilDate::new(2023, 2, 29).unwrap_err(),
            InvalidDate("day is out of range for month".into())
        );
    }

    #[test]
    fn url_fragment_pads_like_glibc_strftime() {
        let d = CivilDate::new(2026, 11, 1).unwrap();
        assert_eq!(d.url_fragment(), "11_01_2026");
        // glibc %Y does not zero-pad: year 9 formats as "9".
        let d = CivilDate::new(9, 1, 5).unwrap();
        assert_eq!(d.url_fragment(), "01_05_9");
        let d = CivilDate::new(999, 12, 31).unwrap();
        assert_eq!(d.url_fragment(), "12_31_999");
    }
}
