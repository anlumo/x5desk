// x5desk's app icon: two fisheye lens discs on a graphite body. The front lens (cool coating) faces the viewer; the
// back lens (warm coating) is turned away, foreshortened and partly hidden: the two halves of a 360-degree capture.
// CoreGraphics only, sRGB output, on the macOS icon grid (824 px body in 1024). A build tool, not run by cargo (needs
// Xcode's actool); what it writes is checked in:
//
//   cd crates/app/icon && swift icon.swift
//
// For the bundle (cargo-packager's `resources` and `icons`, Info.plist's CFBundleIconName): Assets.car, every size from
// 16 to 1024, and AppIcon.icns, 16 and 128 px, for what reads no catalog (the dmg's volume). macOS 27 puts an icon
// that only comes as an icns on a grey plate at 16 and 32 px (Finder's lists, on a 1x screen), as it does Slack's and
// Obsidian's. x5desk.png, 256 px: the window icon where the platform takes one, and a bare binary's Dock icon
// (main.rs).

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

let S: CGFloat = 1024
let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

// MARK: - helpers

func rgb(_ hex: UInt32, _ a: CGFloat = 1) -> CGColor {
    CGColor(colorSpace: sRGB, components: [
        CGFloat((hex >> 16) & 0xff) / 255,
        CGFloat((hex >> 8) & 0xff) / 255,
        CGFloat(hex & 0xff) / 255, a])!
}

func grad(_ stops: [(CGFloat, CGColor)]) -> CGGradient {
    CGGradient(colorsSpace: sRGB, colors: stops.map { $0.1 } as CFArray, locations: stops.map { $0.0 })!
}

func circle(_ c: CGPoint, _ r: CGFloat, _ t: CGAffineTransform = .identity) -> CGPath {
    var t = t
    return CGPath(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r), transform: &t)
}

func ring(_ c: CGPoint, _ r0: CGFloat, _ r1: CGFloat) -> CGPath {
    let p = CGMutablePath()
    p.addPath(circle(c, r1))
    p.addPath(circle(c, r0))
    return p
}

func pt(_ deg: CGFloat, _ r: CGFloat, from c: CGPoint = .zero) -> CGPoint {
    let a = deg * .pi / 180
    return CGPoint(x: c.x + cos(a) * r, y: c.y + sin(a) * r)
}

func makeContext(_ w: Int, _ h: Int) -> CGContext {
    let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: sRGB,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.setAllowsAntialiasing(true)
    ctx.setShouldAntialias(true)
    ctx.interpolationQuality = .high
    return ctx
}

/// CGContextDrawConicGradient has no Swift method spelling in this SDK.
func conic(_ ctx: CGContext, _ g: CGGradient, center: CGPoint, angle: CGFloat) {
    CGContextDrawConicGradient(ctx, g, center, angle)
}

func savePNG(_ img: CGImage, _ path: String) {
    let url = URL(fileURLWithPath: path) as CFURL
    let dst = CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, img, nil)
    guard CGImageDestinationFinalize(dst) else { fatalError("could not write \(path)") }
}

/// Apple-style continuous-corner rounded rect (the "squircle" of the macOS icon grid).
func continuousRect(_ rect: CGRect, radius: CGFloat) -> CGPath {
    let r = min(radius, min(rect.width, rect.height) / 2 / 1.52866483)
    // One corner as (a, b) offsets in units of r: a along the incoming edge, b along the outgoing one.
    let start: (CGFloat, CGFloat) = (1.52866483, 0)
    let curves: [((CGFloat, CGFloat), (CGFloat, CGFloat), (CGFloat, CGFloat))] = [
        ((1.08849323, 0), (0.86840689, 0), (0.63149399, 0.07491100)),
        ((0.37282392, 0.16905899), (0.16905899, 0.37282392), (0.07491100, 0.63149399)),
        ((0, 0.86840701), (0, 1.08849323), (0, 1.52866483)),
    ]
    let (x0, x1, y0, y1) = (rect.minX, rect.maxX, rect.minY, rect.maxY)
    let maps: [(CGFloat, CGFloat) -> CGPoint] = [
        { a, b in CGPoint(x: x1 - a * r, y: y0 + b * r) },
        { a, b in CGPoint(x: x1 - b * r, y: y1 - a * r) },
        { a, b in CGPoint(x: x0 + a * r, y: y1 - b * r) },
        { a, b in CGPoint(x: x0 + b * r, y: y0 + a * r) },
    ]
    let p = CGMutablePath()
    for (i, m) in maps.enumerated() {
        let s = m(start.0, start.1)
        if i == 0 { p.move(to: s) } else { p.addLine(to: s) }
        for (c1, c2, e) in curves {
            p.addCurve(to: m(e.0, e.1), control1: m(c1.0, c1.1), control2: m(c2.0, c2.1))
        }
    }
    p.closeSubpath()
    return p
}

// MARK: - design parameters

/// Light comes from the top left everywhere (degrees, 0 = +x, counter-clockwise, y up).
let lightAngle: CGFloat = 125
let body = CGRect(x: 100, y: 100, width: 824, height: 824)
let bodyPath = continuousRect(body, radius: 185)

struct Lens {
    var center: CGPoint
    var radius: CGFloat
    var coating: [(CGFloat, CGColor)]   // conic stops of the coloured coating reflection
    var coatingAngle: CGFloat           // radians, rotation of the conic gradient
    var glassTint: UInt32
    var squash: CGFloat = 1             // < 1: disc turned away along `axis`, foreshortened
    var axis: CGFloat = 45              // degrees, direction the turned disc faces (in the picture plane)
    var innerShift: CGFloat = 0         // how far the protruding glass slides toward `axis`, in units of radius

    /// Local lens space (centre at origin, unit = 1 px of an unturned lens) to canvas.
    var transform: CGAffineTransform {
        let a = axis * .pi / 180
        return CGAffineTransform(rotationAngle: -a)
            .concatenating(CGAffineTransform(scaleX: squash, y: 1))
            .concatenating(CGAffineTransform(rotationAngle: a))
            .concatenating(CGAffineTransform(translationX: center.x, y: center.y))
    }
    var outline: CGPath { circle(.zero, radius, transform) }
}

// The back lens sits up and to the right along the 45-degree diagonal, turned toward the top right; the
// front lens is turned slightly the other way, so the two visibly face apart.
let centreDistance: CGFloat = 322
let opticalOffset = CGPoint(x: -2, y: 4)   // nudge up-left against the down-right shadows

var front = Lens(
    center: .zero, radius: 236,
    coating: [
        (0.00, rgb(0x2EEBCB)), (0.22, rgb(0x2DB0FF)), (0.48, rgb(0x5D5BFF)),
        (0.72, rgb(0x2DB0FF)), (1.00, rgb(0x2EEBCB)),
    ],
    coatingAngle: 0.6, glassTint: 0x0B1B36,
    squash: 0.95, axis: 225, innerShift: 0.035)

var back = Lens(
    center: pt(45, centreDistance), radius: 190,
    coating: [
        (0.00, rgb(0xFF4F93)), (0.30, rgb(0xBC56FF)), (0.55, rgb(0xFF7158)),
        (0.80, rgb(0xBC56FF)), (1.00, rgb(0xFF4F93)),
    ],
    coatingAngle: 2.2, glassTint: 0x220D26,
    squash: 0.86, axis: 45, innerShift: 0.07)

// Centre the pair's bounding box on the body.
do {
    let bb = front.outline.boundingBoxOfPath.union(back.outline.boundingBoxOfPath)
    let dx = 512 - bb.midX + opticalOffset.x, dy = 512 - bb.midY + opticalOffset.y
    front.center = CGPoint(x: front.center.x + dx, y: front.center.y + dy)
    back.center = CGPoint(x: back.center.x + dx, y: back.center.y + dy)
}

// MARK: - body

func drawBody(_ ctx: CGContext) {
    // Drop shadow inside the margin: a wide soft one and a tight contact one.
    for (dy, blur, a) in [(CGFloat(-14), CGFloat(34), CGFloat(0.42)), (-3, 6, 0.30)] {
        ctx.saveGState()
        ctx.setShadow(offset: CGSize(width: 0, height: dy), blur: blur, color: rgb(0x000000, a))
        ctx.addPath(bodyPath); ctx.setFillColor(rgb(0x2A2C31)); ctx.fillPath()
        ctx.restoreGState()
    }

    // Graphite fill with a soft sheen from the top left.
    ctx.saveGState()
    ctx.addPath(bodyPath); ctx.clip()
    ctx.drawLinearGradient(grad([(0, rgb(0x595E68)), (0.45, rgb(0x373A41)), (1, rgb(0x212328))]),
                           start: CGPoint(x: 0, y: body.maxY), end: CGPoint(x: 0, y: body.minY), options: [])
    ctx.drawRadialGradient(grad([(0, rgb(0xFFFFFF, 0.09)), (1, rgb(0xFFFFFF, 0))]),
                           startCenter: CGPoint(x: 300, y: 880), startRadius: 0,
                           endCenter: CGPoint(x: 300, y: 880), endRadius: 640, options: [])
    ctx.restoreGState()

    // Edge: bright along the top, a faint light line along the bottom so it holds on dark backgrounds.
    ctx.saveGState()
    ctx.addPath(bodyPath); ctx.clip()
    ctx.addPath(bodyPath); ctx.setLineWidth(9); ctx.replacePathWithStrokedPath(); ctx.clip()
    ctx.drawLinearGradient(grad([(0, rgb(0xFFFFFF, 0.34)), (0.5, rgb(0xFFFFFF, 0.08)), (1, rgb(0xFFFFFF, 0.10))]),
                           start: CGPoint(x: 0, y: body.maxY), end: CGPoint(x: 0, y: body.minY), options: [])
    ctx.restoreGState()
}

// MARK: - lens

func drawLens(_ ctx: CGContext, _ L: Lens) {
    let R = L.radius
    let o = CGPoint.zero
    // Light direction in lens space (the squash is mild, so the canvas direction is close enough).
    let tl = pt(lightAngle, R), br = pt(lightAngle + 180, R)

    // Shadow on whatever lies below, cast down and to the right.
    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 12, height: -20), blur: 40, color: rgb(0x000000, 0.65))
    ctx.addPath(L.outline); ctx.setFillColor(rgb(0x101114)); ctx.fillPath()
    ctx.restoreGState()

    ctx.saveGState()
    ctx.concatenate(L.transform)

    // Outer housing: brushed metal ring.
    ctx.saveGState()
    ctx.addPath(circle(o, R)); ctx.clip()
    ctx.drawLinearGradient(grad([(0, rgb(0xDDE1E7)), (0.45, rgb(0x8A8F98)), (1, rgb(0x50545C))]),
                           start: tl, end: br, options: [])
    ctx.setBlendMode(.overlay)
    conic(ctx, grad([
        (0.00, rgb(0xFFFFFF, 0.0)), (0.12, rgb(0xFFFFFF, 0.35)), (0.25, rgb(0x000000, 0.22)),
        (0.37, rgb(0xFFFFFF, 0.30)), (0.50, rgb(0xFFFFFF, 0.0)), (0.62, rgb(0xFFFFFF, 0.30)),
        (0.75, rgb(0x000000, 0.22)), (0.87, rgb(0xFFFFFF, 0.35)), (1.00, rgb(0xFFFFFF, 0.0)),
    ]), center: o, angle: 0.3)
    ctx.restoreGState()

    // Crisp highlight on the outer edge, brightest toward the light, with a faint bounce light on the
    // shadow side so the ring keeps its outline against the dark body.
    ctx.saveGState()
    ctx.addPath(circle(o, R - 2.5)); ctx.setLineWidth(5); ctx.replacePathWithStrokedPath(); ctx.clip()
    ctx.drawLinearGradient(grad([(0, rgb(0xFFFFFF, 0.9)), (0.5, rgb(0xFFFFFF, 0.08)), (0.82, rgb(0xFFFFFF, 0.06)),
                                 (1, rgb(0xFFFFFF, 0.30))]),
                           start: tl, end: br, options: [])
    ctx.restoreGState()

    // Everything inside the housing slides toward the facing direction when the disc is turned.
    let shift = pt(L.axis, R * L.innerShift)
    let shiftGlass = pt(L.axis, R * L.innerShift * 1.6)

    // Black groove between housing and barrel.
    let rGroove = R * 0.845
    ctx.addPath(circle(CGPoint(x: shift.x * 0.6, y: shift.y * 0.6), rGroove)); ctx.setFillColor(rgb(0x08090B)); ctx.fillPath()

    // Barrel: fine concentric rings, lit from the top left.
    let rBarrel = R * 0.818, rGlass = R * 0.675
    ctx.saveGState()
    ctx.addPath(circle(shift, rBarrel)); ctx.clip()
    ctx.drawLinearGradient(grad([(0, rgb(0x41454D)), (1, rgb(0x17191C))]), start: tl, end: br, options: [])
    let steps = 6
    for i in 1...steps {
        let f = CGFloat(i) / CGFloat(steps)
        let r = rGlass + (rBarrel - rGlass) * f
        let c = CGPoint(x: shift.x + (shiftGlass.x - shift.x) * (1 - f), y: shift.y + (shiftGlass.y - shift.y) * (1 - f))
        ctx.addPath(circle(c, r)); ctx.setLineWidth(R * 0.007); ctx.setStrokeColor(rgb(0x000000, 0.6)); ctx.strokePath()
        ctx.addPath(circle(CGPoint(x: c.x + cos(lightAngle * .pi / 180) * 1.5, y: c.y + sin(lightAngle * .pi / 180) * 1.5), r - R * 0.009))
        ctx.setLineWidth(R * 0.004); ctx.setStrokeColor(rgb(0xFFFFFF, 0.11)); ctx.strokePath()
    }
    ctx.restoreGState()

    // Thin lip where the glass meets the barrel: dark toward the light, bright on the far side.
    ctx.saveGState()
    ctx.addPath(circle(shiftGlass, rGlass + R * 0.012)); ctx.setLineWidth(R * 0.022); ctx.replacePathWithStrokedPath(); ctx.clip()
    ctx.drawLinearGradient(grad([(0, rgb(0x000000, 0.9)), (1, rgb(0xFFFFFF, 0.40))]), start: tl, end: br, options: [])
    ctx.restoreGState()

    // Glass dome.
    let g = shiftGlass
    ctx.saveGState()
    ctx.addPath(circle(g, rGlass)); ctx.clip()
    let gc = pt(lightAngle, rGlass * 0.15, from: g)
    ctx.drawRadialGradient(grad([(0, rgb(L.glassTint)), (1, rgb(0x020203))]),
                           startCenter: gc, startRadius: 0, endCenter: g, endRadius: rGlass, options: [])

    // Coloured anti-reflection coating: conic colours shaped into a ring by a dark radial mask.
    conic(ctx, grad(L.coating), center: g, angle: L.coatingAngle)
    ctx.drawRadialGradient(grad([
        (0.00, rgb(0x010103, 0.97)), (0.20, rgb(0x010103, 0.95)), (0.33, rgb(0x010103, 0.22)),
        (0.53, rgb(0x010103, 0.0)), (0.70, rgb(0x010103, 0.32)), (0.88, rgb(0x010103, 0.90)),
        (1.00, rgb(0x010103, 0.98)),
    ]), startCenter: g, startRadius: 0, endCenter: g, endRadius: rGlass, options: [])

    // A smaller lens element seen through the dome.
    ctx.addPath(circle(g, rGlass * 0.22)); ctx.setLineWidth(R * 0.009); ctx.setStrokeColor(rgb(0xFFFFFF, 0.12)); ctx.strokePath()

    // Specular window reflection: a crescent on the lit side.
    let ld = pt(lightAngle, 1)
    let crescent = CGMutablePath()
    crescent.addPath(circle(g, rGlass * 0.92))
    crescent.addPath(circle(CGPoint(x: g.x - ld.x * rGlass * 0.26, y: g.y - ld.y * rGlass * 0.26), rGlass * 0.92))
    ctx.saveGState()
    ctx.addPath(crescent); ctx.clip(using: .evenOdd)
    ctx.addPath(circle(g, rGlass * 0.92)); ctx.clip()
    ctx.drawLinearGradient(grad([(0, rgb(0xFFFFFF, 0.80)), (0.35, rgb(0xFFFFFF, 0.20)), (0.6, rgb(0xFFFFFF, 0))]),
                           start: pt(lightAngle, rGlass, from: g), end: pt(lightAngle + 180, rGlass * 0.4, from: g), options: [])
    ctx.restoreGState()

    // Sparkle and a faint counter-reflection on the far side.
    let sp = pt(lightAngle + 4, rGlass * 0.50, from: g)
    ctx.drawRadialGradient(grad([(0, rgb(0xFFFFFF, 0.95)), (0.45, rgb(0xFFFFFF, 0.6)), (1, rgb(0xFFFFFF, 0))]),
                           startCenter: sp, startRadius: 0, endCenter: sp, endRadius: rGlass * 0.12, options: [])
    let cr = pt(lightAngle + 180, rGlass * 0.62, from: g)
    ctx.drawRadialGradient(grad([(0, rgb(0xFFFFFF, 0.20)), (1, rgb(0xFFFFFF, 0))]),
                           startCenter: cr, startRadius: 0, endCenter: cr, endRadius: rGlass * 0.35, options: [])
    ctx.restoreGState()

    ctx.restoreGState()
}

// MARK: - render

func renderMaster() -> CGImage {
    let ctx = makeContext(Int(S), Int(S))
    ctx.clear(CGRect(x: 0, y: 0, width: S, height: S))
    drawBody(ctx)
    ctx.saveGState()
    ctx.addPath(bodyPath); ctx.clip()
    drawLens(ctx, back)
    // Contact shadow where the front lens overlaps the back one: a short hard core (keeps the discs apart
    // at small sizes) fading out softly (no hard wedge where the two outlines meet).
    ctx.saveGState()
    ctx.addPath(back.outline); ctx.clip()
    ctx.concatenate(front.transform)
    ctx.drawRadialGradient(grad([(0, rgb(0x050607, 0.97)), (0.22, rgb(0x050607, 0.90)), (0.55, rgb(0x050607, 0.45)),
                                 (1, rgb(0x050607, 0))]),
                           startCenter: .zero, startRadius: front.radius - 2, endCenter: .zero,
                           endRadius: front.radius + 34, options: [.drawsBeforeStartLocation])
    ctx.restoreGState()
    drawLens(ctx, front)
    ctx.restoreGState()
    return ctx.makeImage()!
}

/// Downscale by repeated halving with high-quality interpolation (a good box-like filter).
func downscale(_ img: CGImage, to size: Int) -> CGImage {
    var cur = img
    var w = img.width
    while w / 2 >= size {
        w /= 2
        let ctx = makeContext(w, w)
        ctx.draw(cur, in: CGRect(x: 0, y: 0, width: w, height: w))
        cur = ctx.makeImage()!
    }
    if w != size {
        let ctx = makeContext(size, size)
        ctx.draw(cur, in: CGRect(x: 0, y: 0, width: size, height: size))
        cur = ctx.makeImage()!
    }
    return cur
}

let out = FileManager.default.currentDirectoryPath
let master = renderMaster()
// An asset catalog of the sizes for actool, as Xcode builds an app's icon.
let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("x5desk-icon-\(getpid())").path
let cat = tmp + "/x5desk.xcassets", set = cat + "/AppIcon.appiconset"
try! FileManager.default.createDirectory(atPath: set, withIntermediateDirectories: true)
var images: [[String: String]] = []
for pt in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let name = "icon_\(pt)x\(pt)\(scale == 2 ? "@2x" : "").png"
        savePNG(downscale(master, to: pt * scale), "\(set)/\(name)")
        images.append(["idiom": "mac", "size": "\(pt)x\(pt)", "scale": "\(scale)x", "filename": name])
    }
}
let info: [String: Any] = ["author": "xcode", "version": 1]
try! JSONSerialization.data(withJSONObject: ["images": images, "info": info])
    .write(to: URL(fileURLWithPath: set + "/Contents.json"))
try! JSONSerialization.data(withJSONObject: ["info": info]).write(to: URL(fileURLWithPath: cat + "/Contents.json"))
// the deployment target is the bundle's (minimum-system-version)
let actool = try! Process.run(URL(fileURLWithPath: "/usr/bin/xcrun"), arguments: [
    "actool", "--compile", out, "--platform", "macosx", "--minimum-deployment-target", "15.0", "--app-icon", "AppIcon",
    "--output-partial-info-plist", tmp + "/partial.plist", "--output-format", "human-readable-text", cat])
actool.waitUntilExit()
guard actool.terminationStatus == 0 else { fatalError("actool failed") }
try! FileManager.default.removeItem(atPath: tmp)
savePNG(downscale(master, to: 256), out + "/x5desk.png")
print("wrote \(out)/Assets.car, AppIcon.icns, x5desk.png")
