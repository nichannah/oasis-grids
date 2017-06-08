
program atm

use coupler, only : coupler_init, coupler_init_done, coupler_add_field, &
                    coupler_destroy_field, coupler_put, coupler_close, &
                    coupler_destroy_field, couple_field_type, &
                    COUPLER_OUT, coupler_dump_field
implicit none

    type(couple_field_type), dimension(:), allocatable :: fields
    integer :: i, j, t, timestep
    logical :: debug
    real :: time_start, time_finish

    debug = .true.
    timestep = 1

    ! Initialise the coupler.
    call cpu_time(time_start)
    call coupler_init('atmxxx', 192, 94, 1, 1)

    ! Create/add the coupling field.
    allocate(fields(1))
    call coupler_add_field(fields(1), 'src_field', COUPLER_OUT)
    call coupler_init_done()
    call cpu_time(time_finish)

    print*, "Atm startup time: ", time_finish - time_start

    call cpu_time(time_start)
    if (debug) then
      call coupler_dump_field(fields(1), 'src_field.nc')
    endif

    do t=1,1
      fields(1)%field(:, :) = t
      call coupler_put(timestep*t, fields)
    enddo

    call coupler_destroy_field(fields(1))
    call coupler_close()
    call cpu_time(time_finish)

    print*, "Atm runtime: ", time_finish - time_start

    deallocate(fields)

end program atm
