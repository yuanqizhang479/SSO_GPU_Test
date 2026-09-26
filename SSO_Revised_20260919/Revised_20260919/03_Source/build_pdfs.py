#!/usr/bin/env python3
"""Build the shared manuscript and supplement in SIAM or portable article format."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--format', choices=('siam', 'review'), default='siam',
                    help='siam: official class; review: portable article, same body')
    ap.add_argument('--template-dir', type=Path,
                    help='Directory containing a separately installed SIAM macro distribution')
    args = ap.parse_args()
    for executable in ('pdflatex', 'bibtex'):
        if not shutil.which(executable):
            raise SystemExit('Missing %s. Install TeX Live with latex-extra, fonts-recommended, and science (or equivalent MiKTeX packages).' % executable)
    env = os.environ.copy()
    if args.format == 'siam':
        template = (args.template_dir or ROOT / 'vendor' / 'siam_template').resolve()
        classes = list(template.rglob('siamart251216.cls')) if template.exists() else []
        styles = list(template.rglob('siamplain.bst')) if template.exists() else []
        if not classes or not styles:
            raise SystemExit('SIAM macros are missing. Run python download_siam_template.py, or supply --template-dir PATH.\nPortable alternative: python build_pdfs.py --format review')
        env['TEXINPUTS'] = str(classes[0].parent) + os.pathsep + env.get('TEXINPUTS', '')
        env['BSTINPUTS'] = str(styles[0].parent) + os.pathsep + env.get('BSTINPUTS', '')
    suffix = '' if args.format == 'siam' else '_review'
    folder = ROOT / 'submission'
    for name in ('main' + suffix, 'supplement' + suffix):
        # Rebuild disposable intermediates; recover cleanly after interrupted TeX.
        for extension in ('.aux', '.out', '.thm', '.bbl', '.blg'):
            (folder / (name + extension)).unlink(missing_ok=True)
        commands = [
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
            ['bibtex', name],
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
        ]
        logfile = folder / (name + '_build.log')
        with logfile.open('w') as handle:
            for command in commands:
                handle.write('$ ' + ' '.join(command) + '\n'); handle.flush()
                result = subprocess.run(command, cwd=folder, env=env, stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise SystemExit('Build failed; see ' + str(logfile))
        final_log = (folder / (name + '.log')).read_text(errors='replace')
        for _ in range(3):
            if 'Label(s) may have changed' not in final_log:
                break
            with logfile.open('a') as handle:
                result = subprocess.run(commands[-1], cwd=folder, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                raise SystemExit('Build failed; see ' + str(logfile))
            final_log = (folder / (name + '.log')).read_text(errors='replace')
        if 'There were undefined references' in final_log or 'Citation `' in final_log:
            raise SystemExit('Unresolved references/citations; see ' + str(logfile))
        print('Built', folder / (name + '.pdf'))
    print('The review format contains the same body but has different layout, bibliography style, and pagination.')

if __name__ == '__main__':
    main()
