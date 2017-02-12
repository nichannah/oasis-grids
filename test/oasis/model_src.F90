
program model_src

use coupler, only : coupler_init, coupler_init_done, coupler_add_field, &
                    coupler_destroy_field, coupler_put, coupler_close, &
                    coupler_destroy_field, couple_field_type, COUPLER_OUT

implicit none

    type(couple_field_type), dimension(:), allocatable :: fields

    ! Initialise the coupler.
    call coupler_init('srcxxx', 192, 94, 1, 1)

    allocate(fields(1))

    ! Create/add the coupling field.
    call coupler_add_field(fields(1), 'field', COUPLER_OUT)
    call coupler_init_done()

    call coupler_put(0, fields)

    call coupler_destroy_field(fields(1))
    call coupler_close()

end program model_src
