from visualisation import *

from reconstruction import *

# данные для визуализации
full_traj_cm_true = None  # (NPRE+N_STEPS, 3)
full_traj_vertices_true = None  # (NPRE+N_STEPS, 8, 3)
traj_cm_est = None  # (NPRE+N_STEPS, 3)
traj_vertices_est = None  # (NPRE+N_STEPS, 8, 3)
cm_obs = None  # (N_STEPS, 3) - центр масс наблюдений
obs_vertices = None  # (N_STEPS, 8, 3) - наблюдаемые вершины

total_steps = None
frame_idx = 0

# камера и режим просмотра
cam_distance = 60.0
cam_theta = 45.0
cam_phi = 30.0

# режимы:
# 1: точки наблюдений вершин (синие)
# 2: траектории центра масс (true/est) + CM наблюдений
# 3: траектории вершин (true/est)
# 4: пустое поле (только сетка)
# 5: траектории вершин (true/est), как 3
RANDOM_COLOR = False
view_mode = 3
prev_utf8_byte = None  # для отслеживания русской буквы "л"/"Л"

# список «летящих» кубов, каждый хранит:
# {"idx": int, "R_color": np.ndarray (3x3)}
flying_cubes: list[dict] = []


def display():
    global frame_idx, total_steps, RANDOM_COLOR
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    theta_rad = np.radians(cam_theta)
    phi_rad = np.radians(cam_phi)
    cx = cam_distance * np.cos(phi_rad) * np.cos(theta_rad)
    cy = cam_distance * np.sin(phi_rad)
    cz = cam_distance * np.cos(phi_rad) * np.sin(theta_rad)

    gluLookAt(cx, cy, cz, 0.0, 10.0, 0.0, 0.0, 1.0, 0.0)

    draw_grid(size=100.0, step=1.0, y=0.0)

    # 1: только точки наблюдений вершин (синие)
    if view_mode == 1:
        if obs_vertices is not None:
            for i in range(8):
                draw_points(obs_vertices[:, i, :], color=(0.0, 0.5, 1.0), size=4.0)

    # 2: траектории центра масс (истинная и восстановленная) + CM наблюдений
    elif view_mode == 2:
        if full_traj_cm_true is not None:
            draw_trajectory(full_traj_cm_true, color=(0.0, 1.0, 0.0), width=1.5)
        if traj_cm_est is not None:
            draw_trajectory(traj_cm_est, color=(1.0, 0.3, 0.0), width=1.5)

    # 3: траектории вершин (истинные и восстановленные)
    elif view_mode == 3:
        if full_traj_vertices_true is not None:
            for i in range(8):
                draw_trajectory(
                    full_traj_vertices_true[:, i, :], color=(0.0, 0.9, 0.0), width=1.0
                )
        if traj_vertices_est is not None:
            for i in range(8):
                draw_trajectory(
                    traj_vertices_est[:, i, :], color=(1.0, 0.4, 0.0), width=1.0
                )

    # 4: пустое поле (только сетка)
    elif view_mode == 4:
        pass

    # Летящие кубы: каждый идёт по восстановленной траектории с собственной цветовой ориентацией
    if traj_vertices_est is not None and total_steps is not None:
        for cube in flying_cubes:
            k_local = cube["idx"]
            if 0 <= k_local < total_steps:
                verts_k = traj_vertices_est[k_local]
                if RANDOM_COLOR:
                    R_color = cube["R_color"]
                    draw_cube_from_vertices(verts_k, R_color=R_color)
                else:
                    draw_cube_from_vertices(verts_k)

    w = glutGet(GLUT_WINDOW_WIDTH)
    h = glutGet(GLUT_WINDOW_HEIGHT)
    draw_legend(w, h, view_mode, flying_cubes)

    glutSwapBuffers()


def keyboard(key, x, y):
    global cam_distance, view_mode, flying_cubes, prev_utf8_byte

    if key in (b"-", b"_"):
        cam_distance += 2.0
        prev_utf8_byte = None
    elif key in (b"=", b"+"):
        cam_distance = max(5.0, cam_distance - 2.0)
        prev_utf8_byte = None
    elif key == b"1":
        view_mode = 1
        prev_utf8_byte = None
    elif key == b"2":
        view_mode = 2
        prev_utf8_byte = None
    elif key == b"3":
        view_mode = 3
        prev_utf8_byte = None
    elif key == b"4":
        view_mode = 4
        prev_utf8_byte = None
    elif key == b"5":
        view_mode = 5
        prev_utf8_byte = None
    elif key in (b"k", b"K"):
        # запускаем новый куб со старта восстановленной траектории
        if (
            traj_vertices_est is not None
            and total_steps is not None
            and total_steps > 0
        ):
            R_color = random_orthonormal_matrix()
            flying_cubes.append({"idx": 0, "R_color": R_color})
        prev_utf8_byte = None
    else:
        # Обработка русской "л"/"Л" в UTF‑8: D0 BB / D0 9B
        bval = key[0]
        if prev_utf8_byte == 0xD0 and bval in (0xBB, 0x9B):
            if (
                traj_vertices_est is not None
                and total_steps is not None
                and total_steps > 0
            ):
                R_color = random_orthonormal_matrix()
                flying_cubes.append({"idx": 0, "R_color": R_color})
            prev_utf8_byte = None
        else:
            prev_utf8_byte = bval

    glutPostRedisplay()


def special_keys(key, x, y):
    global cam_theta, cam_phi
    if key == GLUT_KEY_LEFT:
        cam_theta += 5.0
    elif key == GLUT_KEY_RIGHT:
        cam_theta -= 5.0
    elif key == GLUT_KEY_UP:
        cam_phi = min(85.0, cam_phi + 3.0)
    elif key == GLUT_KEY_DOWN:
        cam_phi = max(-5.0, cam_phi - 3.0)
    glutPostRedisplay()


def timer(value):
    global frame_idx, total_steps, flying_cubes
    if total_steps is not None and total_steps > 0:
        frame_idx = (frame_idx + 1) % total_steps

        new_list = []
        for cube in flying_cubes:
            cube["idx"] += 1
            if cube["idx"] < total_steps:
                new_list.append(cube)
        flying_cubes = new_list

    glutPostRedisplay()
    glutTimerFunc(16, timer, 0)


def init_glut():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(900, 700)
    glutCreateWindow(b"Inverse problem: full cube trajectory")
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.05, 0.05, 0.08, 1.0)

    # сглаживание линий и точек
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(special_keys)
    glutTimerFunc(0, timer, 0)


def fmt_vec(v, prec=6):
    return "  ".join(f"{x:.{prec}f}" for x in np.asarray(v))


def main():
    global full_traj_cm_true, full_traj_vertices_true
    global traj_cm_est, traj_vertices_est, total_steps, cm_obs, obs_vertices, view_mode

    np.random.seed(0)

    print("Генерируем синтетические данные по истинной модели...")
    full_traj_cm_true, full_traj_vertices_true, V_obs = generate_synthetic_vertex_data(
        dt=DT, n_steps=N_STEPS, npre=NPRE, noise_std_vertices=NOISE_STD_VERTICES
    )

    obs_vertices = V_obs
    cm_obs = compute_com_traj_from_vertices(V_obs)

    total_steps = NPRE + N_STEPS

    print("Запускаем обратную задачу по траектории вершин...")
    result = inverse_from_vertex_trajectories(V_obs)

    p_est = result["p_est"]
    size_est = result["size_est"]
    q_init = result["q_init"]
    L_init = result["L_init"]
    traj_cm_est = result["traj_cm_est"]
    traj_vertices_est = result["traj_vertices_est"]

    I_true = ((2.0 * TRUE_SIZE) ** 2) / 6.0
    L_true = I_true * TRUE_OMEGA

    print("\nИстинные параметры тела в начале эксперимента:")
    print(f"  r0_true   = {fmt_vec(TRUE_R0)}")
    print(f"  l0_true   = {fmt_vec(TRUE_L0)}")
    print(f"  q0_true   = {fmt_vec(TRUE_Q0)}")
    print(f"  size_true = {TRUE_SIZE:.6f}")
    print(f"  L_true    = {fmt_vec(L_true)}")

    print("\nВосстановленные параметры тела:")
    print(f"  r0_est    = {fmt_vec(p_est[0:3])}")
    print(f"  l0_est    = {fmt_vec(p_est[3:6])}")
    print(f"  q0_est    = {fmt_vec(q_init)}")
    print(f"  size_est  = {size_est:.6f}")
    print(f"  L_est     = {fmt_vec(L_init)}")

    init_glut()
    glutMainLoop()


if __name__ == "__main__":
    main()
