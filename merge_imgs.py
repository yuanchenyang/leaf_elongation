from PIL import Image
import os
from utils import *

def round_int(s):
    return int(round(float(s),0))

def main():
    parser = get_default_parser()
    parser.add_argument('-s', '--stitching_result',
                        help='Stitching result csv file',
                        default='stitching_result.csv')
    parser.add_argument('-d', '--result_dir',
                        help='Directory to save merged files',
                        default='merged')
    parser.add_argument('-r', '--include_reverse',
                        help='Whether to additionally include img2 on top of img1',
                        action='store_false')
    args = parser.parse_args()
    open_img = lambda i: Image.open(os.path.join(args.dir, i))
    res_dir = os.path.join(args.dir, args.result_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    with open(os.path.join(args.dir, args.stitching_result)) as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header row
        for img1, img2, dx, dy, _ in reader:
            if args.verbose:
                print('Merging:', img1, img2)
            i1, i2 = map(open_img, (img1, img2))
            dx, dy = map(round_int, (dx, dy))
            W, H = i1.size
            new_W, new_H = W + abs(dx), H + abs(dy)
            i1_x = -dx if dx < 0 else 0
            i1_y = -dy if dy < 0 else 0
            i2_x = dx if dx > 0 else 0
            i2_y = dy if dy > 0 else 0
            res = Image.new(mode='RGB', size=(new_W, new_H))
            res.paste(i1, (i1_x, i1_y))
            res.paste(i2, (i2_x, i2_y))
            res_path = os.path.join(
                args.dir,args.result_dir,
                f'{os.path.splitext(img1)[0]}__{os.path.splitext(img2)[0]}.jpg')
            res.save(res_path)
            if args.include_reverse:
                res.paste(i1, (i1_x, i1_y))
                res.save(res_path[:-4] + '_r.jpg')

if __name__=='__main__':
    main()
