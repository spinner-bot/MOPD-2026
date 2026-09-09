#!/usr/bin/env python3
import pymupdf
import sys

try:
    doc = pymupdf.open('2608.05131v2.pdf')
    print(f'Total pages: {doc.page_count}')
    print('=' * 80)

    # Extract all pages
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        print(f'\n--- Page {page_num + 1} ---\n')
        # Write to stdout with utf-8 encoding
        sys.stdout.buffer.write(f'{text}\n'.encode('utf-8'))

    doc.close()
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
