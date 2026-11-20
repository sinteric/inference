// OpenCV/Clipper compatibility algorithms; attribution and licenses: ../NOTICE.
use crate::{Error, Result};
pub type Point = [f32; 2];
/// Clockwise corners: top-left, top-right, bottom-right, bottom-left.
pub type Quad = [Point; 4];

fn valid(points: &[Point]) -> Result<()> {
    crate::check_shape!(
        points
            .iter()
            .flatten()
            .all(|v| v.is_finite() && v.abs() <= i32::MAX as f32 / 4.),
        "nonfinite or out-of-range geometry"
    );
    Ok(())
}
fn cross(o: Point, a: Point, b: Point) -> f64 {
    (a[0] as f64 - o[0] as f64) * (b[1] as f64 - o[1] as f64)
        - (a[1] as f64 - o[1] as f64) * (b[0] as f64 - o[0] as f64)
}
pub fn convex_hull(points: &[Point]) -> Result<Vec<Point>> {
    valid(points)?;
    let mut p = crate::try_vec(points.len())?;
    p.copy_from_slice(points);
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() <= 2 {
        return Ok(p);
    }
    let mut h = Vec::new();
    h.try_reserve_exact(p.len().checked_mul(2).ok_or(Error::SizeOverflow("hull"))?)
        .map_err(|_| Error::Allocation {
            bytes: points.len().saturating_mul(16),
        })?;
    for &v in &p {
        while h.len() >= 2 && cross(h[h.len() - 2], h[h.len() - 1], v) <= 0. {
            h.pop();
        }
        h.push(v);
    }
    let lower = h.len();
    for &v in p[..p.len() - 1].iter().rev() {
        while h.len() > lower && cross(h[h.len() - 2], h[h.len() - 1], v) <= 0. {
            h.pop();
        }
        h.push(v);
    }
    h.pop();
    Ok(h)
}

pub fn order_quad(mut q: Quad) -> Quad {
    q.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    if q[0][1] > q[1][1] {
        q.swap(0, 1);
    }
    if q[2][1] > q[3][1] {
        q.swap(2, 3);
    }
    [q[0], q[2], q[3], q[1]]
}
/// Minimum area rectangle using rotating calipers. Float normalization and
/// center/angle reconstruction follow OpenCV: tiny rounding differences here
/// affect Clipper's integer truncation, and can move a DB box by a whole pixel.
/// Returns None for fewer than three non-collinear points.
pub fn min_area_rect(points: &[Point]) -> Result<Option<(Quad, f32)>> {
    let mut hull = convex_hull(points)?;
    let n = hull.len();
    if n < 3 {
        return Ok(None);
    }
    // Like OpenCV, preserve monotone input indices when a cyclic shift permits
    // it. This also fixes which edge wins equal-area caliper ties.
    let mut indices = crate::try_vec::<usize>(n)?;
    for (i, p) in hull.iter().enumerate() {
        indices[i] = points.iter().position(|x| x == p).unwrap_or(0);
    }
    let rises = (0..n)
        .filter(|&i| indices[i] < indices[(i + 1) % n])
        .count();
    if rises == 1 || rises == n - 1 {
        let start = if rises == n - 1 {
            (0..n).min_by_key(|&i| indices[i])
        } else {
            (0..n).max_by_key(|&i| indices[i])
        }
        .unwrap_or(0);
        hull.rotate_left(start);
    }
    let mut edges = crate::try_vec::<Point>(n)?;
    let mut inv = crate::try_vec::<f32>(n)?;
    let (mut left, mut right, mut bottom, mut top) = (0, 0, 0, 0);
    for i in 0..n {
        if hull[i][0] < hull[left][0] {
            left = i;
        }
        if hull[i][0] > hull[right][0] {
            right = i;
        }
        if hull[i][1] < hull[bottom][1] {
            bottom = i;
        }
        if hull[i][1] > hull[top][1] {
            top = i;
        }
        edges[i] = [
            hull[(i + 1) % n][0] - hull[i][0],
            hull[(i + 1) % n][1] - hull[i][1],
        ];
        inv[i] = (1. / ((edges[i][0] as f64).powi(2) + (edges[i][1] as f64).powi(2)).sqrt()) as f32;
    }
    let mut seq = [bottom, right, top, left];
    let mut area = f32::INFINITY;
    let mut best = None;
    for _ in 0..n {
        let e = seq.map(|i| edges[i]);
        let r = [
            e[0],
            [e[1][1], -e[1][0]],
            [-e[2][0], -e[2][1]],
            [-e[3][1], e[3][0]],
        ];
        let mut lead = 0;
        for i in 1..4 {
            if r[i][1] * r[lead][0] - r[i][0] * r[lead][1] < 0. {
                lead = i;
            }
        }
        let idx = seq[lead];
        let x = edges[idx][0] * inv[idx];
        let y = edges[idx][1] * inv[idx];
        let (a, b) = match lead {
            0 => (x, y),
            1 => (y, -x),
            2 => (-x, -y),
            _ => (-y, x),
        };
        seq[lead] = (seq[lead] + 1) % n;
        let d = [
            hull[seq[1]][0] - hull[seq[3]][0],
            hull[seq[1]][1] - hull[seq[3]][1],
        ];
        let w = d[0] * a + d[1] * b;
        let d = [
            hull[seq[2]][0] - hull[seq[0]][0],
            hull[seq[2]][1] - hull[seq[0]][1],
        ];
        let h = -d[0] * b + d[1] * a;
        if w * h <= area {
            area = w * h;
            best = Some((a, b, w, h, seq[3], seq[0]));
        }
    }
    let Some((a, b, w, h, l, d)) = best else {
        return Ok(None);
    };
    let c1 = a * hull[l][0] + hull[l][1] * b;
    let c2 = -b * hull[d][0] + hull[d][1] * a;
    let determinant = 1. / (a * a + b * b);
    let corner = [
        (c1 * a - c2 * b) * determinant,
        (a * c2 + b * c1) * determinant,
    ];
    let v1 = [a * w, b * w];
    let v2 = [-b * h, a * h];
    let center = [
        corner[0] + (v1[0] + v2[0]) * 0.5,
        corner[1] + (v1[1] + v2[1]) * 0.5,
    ];
    let mut width = ((v2[0] as f64).powi(2) + (v2[1] as f64).powi(2)).sqrt() as f32;
    let mut height = ((v1[0] as f64).powi(2) + (v1[1] as f64).powi(2)).sqrt() as f32;
    let angle = if v1[0] == 0. && v1[1] > 0. {
        std::mem::swap(&mut width, &mut height);
        -std::f64::consts::FRAC_PI_2
    } else {
        -(v1[0] as f64).atan2(v1[1] as f64)
    };
    let degrees = (angle * 180. / std::f64::consts::PI) as f32;
    let radians = degrees as f64 * std::f64::consts::PI / 180.;
    let ca = radians.cos() as f32 * 0.5;
    let sa = radians.sin() as f32 * 0.5;
    let p0 = [
        center[0] - sa * height - ca * width,
        center[1] + ca * height - sa * width,
    ];
    let p1 = [
        center[0] + sa * height - ca * width,
        center[1] - ca * height - sa * width,
    ];
    Ok(Some((
        order_quad([
            p0,
            p1,
            [2. * center[0] - p0[0], 2. * center[1] - p0[1]],
            [2. * center[0] - p1[0], 2. * center[1] - p1[1]],
        ]),
        width.min(height),
    )))
}

/// Suzuki-Abe border following, RETR_LIST / CHAIN_APPROX_SIMPLE. Nonzero mask
/// bytes are foreground. Includes hole borders; list order matches OpenCV's
/// reverse raster discovery order, including contours touching image edges.
pub fn contours(mask: &[u8], width: usize, height: usize) -> Result<Vec<Vec<Point>>> {
    crate::check_shape!(
        width > 0
            && height > 0
            && width <= i32::MAX as usize / 4
            && height <= i32::MAX as usize / 4,
        "invalid contour dimensions"
    );
    crate::check_shape!(
        mask.len() == crate::checked_size("contour mask", &[width, height])?,
        "contour mask extent"
    );
    let stride = width
        .checked_add(2)
        .ok_or(Error::SizeOverflow("contour row"))?;
    let rows = height
        .checked_add(2)
        .ok_or(Error::SizeOverflow("contour rows"))?;
    let mut image = crate::try_vec::<i32>(crate::checked_size("contour image", &[stride, rows])?)?;
    for y in 0..height {
        for x in 0..width {
            image[(y + 1) * stride + x + 1] = i32::from(mask[y * width + x] != 0);
        }
    }
    // Clockwise neighbours, starting east, in image coordinates.
    let offsets = [
        1,
        1 + stride as isize,
        stride as isize,
        stride as isize - 1,
        -1,
        -1 - stride as isize,
        -(stride as isize),
        1 - stride as isize,
    ];
    let xy = |p: usize| [(p % stride - 1) as f32, (p / stride - 1) as f32];
    let mut result = Vec::new();
    let mut border = 1i32;
    for y in 1..=height {
        for x in 1..=width {
            let start = y * stride + x;
            let back = if image[start] == 1 && image[start - 1] == 0 {
                4
            } else if image[start] >= 1 && image[start + 1] == 0 {
                0
            } else {
                continue;
            };
            border = border
                .checked_add(1)
                .ok_or(Error::SizeOverflow("contour labels"))?;
            let first = (0..8)
                .map(|k| (back + k) % 8)
                .find(|&k| image[(start as isize + offsets[k]) as usize] != 0);
            let mut contour = Vec::new();
            // A traced boundary has at most eight directed visits per foreground
            // pixel. Grow outside the inner neighbour search, returning alloc errors.
            if let Some(first) = first {
                let first_pos = (start as isize + offsets[first]) as usize;
                let mut prev = first_pos;
                let mut current = start;
                let mut last_dir = None;
                loop {
                    let back = offsets
                        .iter()
                        .position(|&o| current as isize + o == prev as isize)
                        .ok_or_else(|| Error::Backend("contour predecessor".into()))?;
                    let mut east_zero = false;
                    let dir = (1..=8)
                        .map(|k| (back + 8 - k) % 8)
                        .find(|&k| {
                            let nonzero = image[(current as isize + offsets[k]) as usize] != 0;
                            if k == 0 && !nonzero {
                                east_zero = true;
                            }
                            nonzero
                        })
                        .ok_or_else(|| Error::Backend("contour neighbour".into()))?;
                    let next = (current as isize + offsets[dir]) as usize;
                    if east_zero {
                        image[current] = -border;
                    } else if image[current] == 1 {
                        image[current] = border;
                    }
                    if last_dir != Some(dir) {
                        contour.try_reserve(1).map_err(|_| Error::Allocation {
                            bytes: contour.len().saturating_mul(8),
                        })?;
                        contour.push(xy(current));
                        last_dir = Some(dir);
                    }
                    if next == start && current == first_pos {
                        break;
                    }
                    prev = current;
                    current = next;
                }
                // If the first and final segments share a direction, the start is
                // interior to that segment and CHAIN_APPROX_SIMPLE omits it.
                if contour.len() > 2 {
                    let n = contour.len();
                    if cross(contour[n - 1], contour[0], contour[1]) == 0.
                        && (contour[0][0] - contour[n - 1][0]) * (contour[1][0] - contour[0][0])
                            + (contour[0][1] - contour[n - 1][1]) * (contour[1][1] - contour[0][1])
                            > 0.
                    {
                        contour.remove(0);
                    }
                }
            } else {
                image[start] = -border;
                contour.push(xy(start));
            }
            result.try_reserve(1).map_err(|_| Error::Allocation {
                bytes: result.len().saturating_mul(24),
            })?;
            result.push(contour);
        }
    }
    result.reverse();
    Ok(result)
}

/// Round-join outward offset of a convex polygon using Clipper's integer
/// coordinates, arc tolerance 0.25 and half-away-from-zero rounding. This is
/// sufficient for DB's minimum rectangles; concave input is rejected.
pub fn unclip(q: Quad, ratio: f32) -> Result<Vec<Point>> {
    valid(&q)?;
    crate::check_shape!(ratio.is_finite() && ratio > 0., "invalid unclip ratio");
    let mut area = 0f64;
    let mut perimeter = 0f64;
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        area += a[0] as f64 * b[1] as f64 - b[0] as f64 * a[1] as f64;
        perimeter += (b[0] as f64 - a[0] as f64).hypot(b[1] as f64 - a[1] as f64);
    }
    if perimeter == 0. || area == 0. {
        return Ok(vec![]);
    }
    let delta = area.abs() * 0.5 * ratio as f64 / perimeter;
    crate::check_shape!(
        delta.is_finite() && delta <= i32::MAX as f64 / 8.,
        "unclip distance too large"
    );
    let mut p = q.map(|a| [a[0].trunc(), a[1].trunc()]);
    if area < 0. {
        p.reverse();
    }
    if (0..4).any(|i| cross(p[i], p[(i + 1) % 4], p[(i + 2) % 4]) <= 0.) {
        return Ok(vec![]);
    }
    let normals: [[f64; 2]; 4] = std::array::from_fn(|i| {
        let dx = p[(i + 1) % 4][0] as f64 - p[i][0] as f64;
        let dy = p[(i + 1) % 4][1] as f64 - p[i][1] as f64;
        let l = dx.hypot(dy);
        [dy / l, -dx / l]
    });
    let tolerance = 0.25f64.min(delta * 0.25);
    let steps =
        (std::f64::consts::PI / (1. - tolerance / delta).acos()).min(delta * std::f64::consts::PI);
    let angle = std::f64::consts::TAU / steps;
    let (sin, cos) = angle.sin_cos();
    let capacity = (steps.ceil() as usize)
        .checked_add(8)
        .ok_or(Error::SizeOverflow("unclip points"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| Error::Allocation {
            bytes: capacity.saturating_mul(8),
        })?;
    for j in 0..4 {
        let k = (j + 3) % 4;
        let mut n = normals[k];
        let m = normals[j];
        let a = (n[0] * m[1] - m[0] * n[1])
            .clamp(-1., 1.)
            .atan2(n[0] * m[0] + n[1] * m[1]);
        let count = (steps / std::f64::consts::TAU * a.abs()).round().max(1.) as usize;
        for _ in 0..count {
            out.push([
                (p[j][0] as f64 + n[0] * delta).round() as f32,
                (p[j][1] as f64 + n[1] * delta).round() as f32,
            ]);
            n = [n[0] * cos - n[1] * sin, n[0] * sin + n[1] * cos];
        }
        out.push([
            (p[j][0] as f64 + m[0] * delta).round() as f32,
            (p[j][1] as f64 + m[1] * delta).round() as f32,
        ]);
    }
    Ok(out)
}

// Integer Cohen-Sutherland clipping with OpenCV's truncation convention.
fn clip_line(a: &mut [i64; 2], b: &mut [i64; 2], w: i64, h: i64) -> bool {
    let code = |p: [i64; 2]| {
        i32::from(p[0] < 0)
            + 2 * i32::from(p[0] >= w)
            + 4 * i32::from(p[1] < 0)
            + 8 * i32::from(p[1] >= h)
    };
    let mut ca = code(*a);
    let mut cb = code(*b);
    if ca & cb == 0 && ca | cb != 0 {
        if ca & 12 != 0 {
            let y = if ca < 8 { 0 } else { h - 1 };
            a[0] += ((y - a[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            a[1] = y;
            ca = code(*a) & 3;
        }
        if cb & 12 != 0 {
            let y = if cb < 8 { 0 } else { h - 1 };
            b[0] += ((y - b[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            b[1] = y;
            cb = code(*b) & 3;
        }
        if ca & cb == 0 && ca | cb != 0 {
            if ca != 0 {
                let x = if ca == 1 { 0 } else { w - 1 };
                a[1] += ((x - a[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                a[0] = x;
                ca = 0;
            }
            if cb != 0 {
                let x = if cb == 1 { 0 } else { w - 1 };
                b[1] += ((x - b[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                b[0] = x;
                cb = 0;
            }
        }
    }
    ca | cb == 0
}

/// Mean probability inside a box polygon, including its LINE_8 raster border.
/// The float corners are translated to the clipped ROI and truncated, as in
/// Paddle's box_score_fast + cv2.fillPoly. Values are validated by DB's caller.
pub fn polygon_mean_score(prob: &[f32], width: usize, height: usize, q: Quad) -> Result<f32> {
    valid(&q)?;
    crate::check_shape!(
        width > 0
            && height > 0
            && width <= i32::MAX as usize / 4
            && height <= i32::MAX as usize / 4
            && prob.len() == crate::checked_size("score map", &[width, height])?,
        "score map extent"
    );
    let xmin = q
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0., (width - 1) as f32) as usize;
    let xmax = q
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0., (width - 1) as f32) as usize;
    let ymin = q
        .iter()
        .map(|p| p[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0., (height - 1) as f32) as usize;
    let ymax = q
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0., (height - 1) as f32) as usize;
    let w = xmax - xmin + 1;
    let h = ymax - ymin + 1;
    let mut mask = crate::try_vec::<u8>(crate::checked_size("polygon ROI", &[w, h])?)?;
    let p = q.map(|a| [(a[0] - xmin as f32) as i64, (a[1] - ymin as f32) as i64]);
    // LINE_8 boundary using OpenCV's major-axis half-error convention.
    for i in 0..4 {
        let (mut a, mut b) = (p[i], p[(i + 1) % 4]);
        if !clip_line(&mut a, &mut b, w as i64, h as i64) {
            continue;
        }
        if a[0] > b[0] {
            std::mem::swap(&mut a, &mut b);
        }
        let dx = b[0] - a[0];
        let dy = (b[1] - a[1]).abs();
        let sy = if b[1] >= a[1] { 1 } else { -1 };
        let mut x = a[0];
        let mut y = a[1];
        if dx >= dy {
            let mut err = dx - 2 * dy;
            for _ in 0..=dx {
                if x >= 0 && y >= 0 && x < w as i64 && y < h as i64 {
                    mask[y as usize * w + x as usize] = 1;
                }
                if err < 0 {
                    y += sy;
                    err += 2 * dx;
                }
                x += 1;
                err -= 2 * dy;
            }
        } else {
            let mut err = dy - 2 * dx;
            for _ in 0..=dy {
                if x >= 0 && y >= 0 && x < w as i64 && y < h as i64 {
                    mask[y as usize * w + x as usize] = 1;
                }
                if err < 0 {
                    x += 1;
                    err += 2 * dy;
                }
                y += sy;
                err -= 2 * dx;
            }
        }
    }
    for y in 0..h {
        let mut xs = [0i64; 4];
        let mut n = 0;
        for i in 0..4 {
            let (mut a, mut b) = (p[i], p[(i + 1) % 4]);
            if a[1] > b[1] {
                std::mem::swap(&mut a, &mut b);
            }
            if a[1] <= y as i64 && (y as i64) < b[1] {
                let (mut ca, mut cb) = (a, b);
                clip_line(&mut ca, &mut cb, w as i64, h as i64);
                if ca[1] == cb[1] {
                    ca[1] = a[1];
                    cb[1] = b[1];
                }
                let slope = ((cb[0] - ca[0]) << 16) / (cb[1] - ca[1]);
                xs[n] = (ca[0] << 16) + (y as i64 - ca[1]) * slope;
                n += 1;
            }
        }
        xs[..n].sort_unstable();
        for pair in xs[..n].as_chunks::<2>().0 {
            let l = ((pair[0] + 65535) >> 16).clamp(0, w as i64);
            let r = (pair[1] >> 16).clamp(-1, w as i64 - 1);
            if l <= r {
                mask[y * w + l as usize..=y * w + r as usize].fill(1);
            }
        }
    }
    let mut sum = 0f64;
    let mut count = 0usize;
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] != 0 {
                sum += prob[(y + ymin) * width + x + xmin] as f64;
                count += 1;
            }
        }
    }
    Ok(if count == 0 {
        0.
    } else {
        (sum / count as f64) as f32
    })
}
